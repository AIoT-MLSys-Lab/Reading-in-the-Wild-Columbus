import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import rotate_camera_calib_cw90deg
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4
from torch.utils.data import Dataset

# Import your own modules and constants
from ritw.utils import project_gaze_vrs, create_sampled_array, project_gaze_mp4, get_calib_from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RITWDataHandlerVRS(Dataset):
    def __init__(self, cfg: DictConfig, vrs_file: str):
        """
        Dataset that handles the loading and processing of gaze, IMU, and RGB data from a VRS file.

        Args:
            cfg (DictConfig): Configuration object.
            vrs_file (str): Full path to the VRS file.
        """
        self.cfg = cfg

        # Set modality parameters based on the model name
        self.input_hz = 60
        self.input_sec = 2
        self.crop_size = 64
        if cfg.model_name == 'v1_15Hz':
            self.input_hz = 15
        if cfg.model_name == 'v1_1s':
            self.input_sec = 1
        if cfg.model_name == 'v1_large':
            self.crop_size = 128

        self.input_length = self.input_hz * self.input_sec

        # Load VRS file and derive related paths
        self.vrs_file = Path(vrs_file)
        if not self.vrs_file.exists():
            raise FileNotFoundError(f"Input VRS file not found: {vrs_file}")
        self.vrs_path = str(self.vrs_file)
        self.vid_uid = self.vrs_file.stem

        # Determine gaze file path; check for personalized first, then general
        mps_folder = self.vrs_file.parent / f"mps_{self.vid_uid}_vrs"
        if 'mps_folder' in str(self.cfg) and self.cfg.mps_folder:
            mps_folder = Path(self.cfg.mps_folder) / f"mps_{self.vid_uid}_vrs"
        gaze_path = mps_folder / "eye_gaze" / "personalized_eye_gaze.csv"
        if not gaze_path.exists():
            gaze_path = mps_folder/ "eye_gaze" / "general_eye_gaze.csv"
        if not gaze_path.exists():
            raise FileNotFoundError(f"Gaze CSV file not found in {mps_folder}")

        # Process gaze data
        self.gaze = project_gaze_vrs(str(gaze_path), vrs_path=self.vrs_path)

        # Process IMU data if requested
        self.odometry = None
        if self.cfg.use_imu:
            odom_path = mps_folder / "slam" / "open_loop_trajectory.csv"
            if not odom_path.exists():
                raise FileNotFoundError(f"Odometry CSV file not found: {odom_path}")
            self.odometry = pd.read_csv(odom_path, engine='python')
            self.odometry = self.odometry[[
                "device_linear_velocity_x_odometry",
                "device_linear_velocity_y_odometry",
                "device_linear_velocity_z_odometry",
                "angular_velocity_x_device",
                "device_linear_velocity_y_odometry",  # duplicate column as in original code
                "angular_velocity_z_device"
            ]]
            indices = np.linspace(0, len(self.odometry) - 1, len(self.gaze), dtype=int)
            self.odometry = self.odometry.iloc[indices]

        # Initialize RGB provider if requested
        self.provider = None
        self.gaze_xy = None
        if self.cfg.use_rgb:
            self.provider = data_provider.create_vrs_data_provider(self.vrs_path)
            deliver_option = self.provider.get_default_deliver_queued_options()
            deliver_option.deactivate_stream_all()
            deliver_option.activate_stream(StreamId("214-1"))
            self.gaze_xy = np.array(self.gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())

        # Process input modalities: create a snippet of gaze data and compute differences.
        gaze_data = self.gaze[['transformed_gaze_x', 'transformed_gaze_y', 'transformed_gaze_z']].ffill()
        sampled_gaze = create_sampled_array(gaze_data, num_samples=self.input_length + 1,
                                            stride=60 // self.input_hz)
        # Compute differences (velocity-like features) scaled by input_hz
        self.gaze_sequence = torch.Tensor(np.diff(sampled_gaze, axis=1) * self.input_hz)
        self.num_gaze = self.gaze_sequence.size(0)
        self.gaze_timestamps = self.gaze['tracking_timestamp_us'].tolist()

        if self.cfg.use_imu:
            odom_array = create_sampled_array(self.odometry, num_samples=self.input_length,
                                              stride=60 // self.input_hz)
            self.odometry_sequence = torch.Tensor(odom_array)[:-1]
        else:
            # Even if not used, create a dummy tensor of zeros with the proper shape.
            self.odometry_sequence = torch.zeros((self.num_gaze, self.input_length, 6))

        # Create valid indices for snippet extraction: each snippet uses frames [i, i+input_length]
        start_idx = int(self.cfg.start_time * 60)
        gap = round(self.cfg.snippet_gap * 60)
        self.indices = [i for i in range(start_idx, self.num_gaze, gap)
                        if (i + self.input_length) < len(self.gaze_timestamps)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # Initialize snippet with zeros for each modality.
        snippet = {
            'gaze': torch.zeros((self.input_length, 3), dtype=torch.float32),
            'odom': torch.zeros((self.input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        }

        # Use processed gaze features.
        if self.cfg.use_gaze:
            snippet['gaze'] = self.gaze_sequence[i]

        # Use processed odometry if available.
        if self.cfg.use_imu:
            snippet['odom'] = self.odometry_sequence[i]

        # Use RGB provider to get the image at the end of the snippet.
        if self.cfg.use_rgb and self.provider is not None and self.gaze_xy is not None:
            gaze_idx = i + self.input_length  # Use RGB at the end of the gaze snippet.
            time_ns = self.gaze_timestamps[gaze_idx] * 1000  # Convert microseconds to nanoseconds.
            im_data = self.provider.get_image_data_by_time_ns(
                StreamId("214-1"), time_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
            )[0]
            im = im_data.to_numpy_array()
            # Rotate and convert the image.
            im = cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)
            x_center = 1408 - np.clip(int(self.gaze_xy[gaze_idx, 0]), self.crop_size // 2, 1408 - self.crop_size // 2)
            y_center = np.clip(int(self.gaze_xy[gaze_idx, 1]), self.crop_size // 2, 1408 - self.crop_size // 2)
            gaze_crop = im[y_center - self.crop_size // 2:y_center + self.crop_size // 2, x_center - self.crop_size // 2:x_center + self.crop_size // 2]

            snippet['rgb'] = (torch.Tensor(gaze_crop) / 255.).permute(2, 0, 1)

        # Also return the corresponding timestamp.
        timestamp = self.gaze_timestamps[i + self.input_length]
        return snippet, timestamp


class RITWDataHandlerMP4(Dataset):
    def __init__(self, cfg: DictConfig, mp4_file: str):
        """
        Dataset that handles loading and processing of gaze, IMU, and RGB data from an MP4-exported VRS file.
        Args:
            cfg (DictConfig): Configuration object.
            mp4_file (str): Full path to the MP4 file.
        """
        self.cfg = cfg
        torch.save((self.cfg, mp4_file), 'ritw_data_handler_config.pt')
        self.mp4_file = mp4_file


        # Set parameters based on model
        self.input_hz = 60
        self.input_sec = 2
        self.crop_size = 64
        if cfg.model_name == 'v1_15Hz':
            self.input_hz = 15
        if cfg.model_name == 'v1_1s':
            self.input_sec = 1
        if cfg.model_name == 'v1_large':
            self.crop_size = 128

        self.input_length = self.input_hz * self.input_sec

        # Path setup
        self.mp4_file = Path(mp4_file)
        if not self.mp4_file.exists():
            raise FileNotFoundError(f"Input MP4 file not found: {mp4_file}")
        self.vid_uid = self.mp4_file.stem

        # Gaze CSV path: try personalized, then general
        mps_folder = self.mp4_file.parent / 'mps' / f"mps_{self.vid_uid}_vrs"
        if 'mps_folder' in str(self.cfg) and self.cfg.mps_folder:
            mps_folder = Path(self.cfg.mps_folder) / f"mps_{self.vid_uid}_vrs"
        gaze_path = mps_folder / "eye_gaze" / "personalized_eye_gaze.csv"
        if not gaze_path.exists():
            gaze_path = mps_folder / "eye_gaze" / "general_eye_gaze.csv"
        if not gaze_path.exists():
            raise FileNotFoundError(f"Gaze CSV file not found in {mps_folder}")

        # Load gaze, as before
        calib_path = self.mp4_file.parent.parent / 'calib' / f"{self.vid_uid}.pkl"
        calib, cpf2rgbT = get_calib_from_path(calib_path)
        # calib = rotate_camera_calib_cw90deg(calib)
        self.gaze = project_gaze_mp4(str(gaze_path), calib, cpf2rgbT)

        # Load IMU (odometry) if requested
        self.odometry = None
        if self.cfg.use_imu:
            odom_path =  mps_folder / "slam" / "open_loop_trajectory.csv"
            if not odom_path.exists():
                raise FileNotFoundError(f"Odometry CSV file not found: {odom_path}")
            self.odometry = pd.read_csv(odom_path, engine='python')
            self.odometry = self.odometry[[
                "device_linear_velocity_x_odometry",
                "device_linear_velocity_y_odometry",
                "device_linear_velocity_z_odometry",
                "angular_velocity_x_device",
                "device_linear_velocity_y_odometry",  # duplicate for compatibility
                "angular_velocity_z_device"
            ]]
            indices = np.linspace(0, len(self.odometry) - 1, len(self.gaze), dtype=int)
            self.odometry = self.odometry.iloc[indices]

        # --- MP4 frame & timestamp setup ---
        self.frame_timestamps_ns = get_timestamp_from_mp4(str(self.mp4_file))
        self.cap = cv2.VideoCapture(str(self.mp4_file))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # For gaze-based cropping
        self.gaze_xy = np.array(self.gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())

        # Precompute gaze/odom features
        gaze_data = self.gaze[['transformed_gaze_x', 'transformed_gaze_y', 'transformed_gaze_z']].ffill()
        sampled_gaze = create_sampled_array(gaze_data, num_samples=self.input_length + 1,
                                            stride=60 // self.input_hz)
        self.gaze_sequence = torch.Tensor(np.diff(sampled_gaze, axis=1) * self.input_hz)
        self.num_gaze = self.gaze_sequence.size(0)
        self.gaze_timestamps = self.gaze['tracking_timestamp_us'].tolist()  # Still in microseconds

        if self.cfg.use_imu:
            odom_array = create_sampled_array(self.odometry, num_samples=self.input_length,
                                              stride=60 // self.input_hz)
            self.odometry_sequence = torch.Tensor(odom_array)[:-1]
        else:
            self.odometry_sequence = torch.zeros((self.num_gaze, self.input_length, 6))

        # Create valid snippet indices
        start_idx = int(self.cfg.start_time * 60)
        gap = round(self.cfg.snippet_gap * 60)
        self.indices = [i for i in range(start_idx, self.num_gaze, gap)
                        if (i + self.input_length) < len(self.gaze_timestamps)]

        # Get shape for cropping
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame0 = self.cap.read()
        frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)  # Rotate to match gaze projection
        if not ret:
            raise RuntimeError(f"Could not read first frame from {mp4_file}")
        h, w, _ = frame0.shape
        self.rgb_height, self.rgb_width = w, h  # Since it's rotated

    def __len__(self):
        return len(self.indices)

    def get_frame_by_time_ns(self, target_ns):
        """
        Returns the MP4 frame (after 90deg rotation) closest to the given device_time_ns timestamp.
        """
        idx = np.argmin(np.abs(self.frame_timestamps_ns - target_ns))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {self.mp4_file}")
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def __getitem__(self, idx):
        i = self.indices[idx]
        snippet = {
            'gaze': torch.zeros((self.input_length, 3), dtype=torch.float32),
            'odom': torch.zeros((self.input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        }

        if self.cfg.use_gaze:
            snippet['gaze'] = self.gaze_sequence[i]
        if self.cfg.use_imu:
            snippet['odom'] = self.odometry_sequence[i]

        # --- RGB: get image at end of snippet ---
        if self.cfg.use_rgb:
            gaze_idx = i + self.input_length
            # Timestamp at which we want the image, in ns
            time_ns = self.gaze_timestamps[gaze_idx] * 1000  # microseconds -> nanoseconds
            frame = self.get_frame_by_time_ns(time_ns)
            im = np.rot90(frame, k=-3)
            x_center = 1408 - np.clip(int(self.gaze_xy[gaze_idx, 0]), self.crop_size // 2, 1408 - self.crop_size // 2)
            y_center = np.clip(int(self.gaze_xy[gaze_idx, 1]), self.crop_size // 2, 1408 - self.crop_size // 2)
            gaze_crop = im[y_center - self.crop_size // 2:y_center + self.crop_size // 2, x_center - self.crop_size // 2:x_center + self.crop_size // 2]
            snippet['rgb'] = (torch.from_numpy(gaze_crop.copy()) / 255.).permute(2, 0, 1)

        timestamp = self.gaze_timestamps[i + self.input_length]
        return snippet, timestamp

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()


if __name__ == "__main__":
    import yaml
    cfg = yaml.load(open("/research/nfs_zhang_13664/samiul/ritw-osu/config/predict.yaml", "r"), Loader=yaml.FullLoader)
    cfg['model_name'] = 'v1_default'  # Set a default model name for testing
    cfg['use_rgb'] = True
    cfg['use_gaze'] = True
    cfg['use_imu'] = True
    cfg = DictConfig(cfg)
    dataset = RITWDataHandlerMP4(cfg,
                                 "/research/"
                                 "nfs_zhang_13664/"
                                 "deidentified_IRB_DATA_MP4/mp4/"
                                 "127b188e-9fe3-4c1b-8122-1186175d8723.mp4")
    dataset[200]