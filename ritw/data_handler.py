import logging
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from torch.utils.data import Dataset

# Add your own imports here
from ritw.utils import project_gaze_vrs, create_sampled_array, project_gaze_mp4, get_calib_from_path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseRITWDataHandler(Dataset):
    def __init__(self, cfg: DictConfig, file_path: str, is_vrs: bool):
        self.cfg = cfg
        self.file_path = Path(file_path)
        self.is_vrs = is_vrs
        self._init_params()
        self.vid_uid = self.file_path.stem
        self.mps_folder = self._get_mps_folder()
        self.gaze = self._load_gaze()
        self.gaze_xy = np.array(self.gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())
        self.odometry = self._load_odometry() if self.cfg.use_imu else None
        self.gaze_sequence, self.gaze_timestamps, self.num_gaze = self._prepare_gaze_sequence()
        self.odometry_sequence = self._prepare_odom_sequence() if self.cfg.use_imu else torch.zeros((self.num_gaze, self.input_length, 6))
        self.indices = self._create_indices()
        self.crop_shape = self._get_crop_shape()
        self.rgb_handler = self._init_rgb_handler() if self.cfg.use_rgb else None

    def _init_params(self):
        self.input_hz = 60
        self.input_sec = 2
        self.crop_size = 64
        if getattr(self.cfg, "model_name", None) == "v1_15Hz":
            self.input_hz = 15
        if getattr(self.cfg, "model_name", None) == "v1_1s":
            self.input_sec = 1
        if getattr(self.cfg, "model_name", None) == "v1_large":
            self.crop_size = 128
        self.input_length = self.input_hz * self.input_sec

    def _get_mps_folder(self):
        if "mps_folder" in str(self.cfg) and getattr(self.cfg, "mps_folder", None):
            return Path(self.cfg.mps_folder) / f"mps_{self.vid_uid}_vrs"
        else:
            return self.file_path.parent / (f"mps_{self.vid_uid}_vrs" if self.is_vrs else f"mps/mps_{self.vid_uid}_vrs")

    def _load_gaze(self):
        gaze_path = self.mps_folder / "eye_gaze" / "personalized_eye_gaze.csv"
        if not gaze_path.exists():
            gaze_path = self.mps_folder / "eye_gaze" / "general_eye_gaze.csv"
        if not gaze_path.exists():
            raise FileNotFoundError(f"Gaze CSV file not found in {self.mps_folder}")
        if self.is_vrs:
            return project_gaze_vrs(str(gaze_path), vrs_path=str(self.file_path))
        else:
            calib_path = self.file_path.parent.parent / 'calib' / f"{self.vid_uid}.pkl"
            calib, cpf2rgbT = get_calib_from_path(calib_path)
            return project_gaze_mp4(str(gaze_path), calib, cpf2rgbT)

    def _load_odometry(self):
        odom_path = self.mps_folder / "slam" / "open_loop_trajectory.csv"
        if not odom_path.exists():
            raise FileNotFoundError(f"Odometry CSV file not found: {odom_path}")
        odom = pd.read_csv(odom_path, engine='python')
        odom = odom[[
            "device_linear_velocity_x_odometry",
            "device_linear_velocity_y_odometry",
            "device_linear_velocity_z_odometry",
            "angular_velocity_x_device",
            "device_linear_velocity_y_odometry",  # kept as in original for compat
            "angular_velocity_z_device"
        ]]
        indices = np.linspace(0, len(odom) - 1, len(self.gaze), dtype=int)
        return odom.iloc[indices]

    def _prepare_gaze_sequence(self):
        gaze_data = self.gaze[['transformed_gaze_x', 'transformed_gaze_y', 'transformed_gaze_z']].ffill()
        sampled_gaze = create_sampled_array(gaze_data, num_samples=self.input_length + 1, stride=60 // self.input_hz)
        gaze_sequence = torch.Tensor(np.diff(sampled_gaze, axis=1) * self.input_hz)
        gaze_timestamps = self.gaze['tracking_timestamp_us'].tolist()
        return gaze_sequence, gaze_timestamps, gaze_sequence.size(0)

    def _prepare_odom_sequence(self):
        odom_array = create_sampled_array(self.odometry, num_samples=self.input_length, stride=60 // self.input_hz)
        return torch.Tensor(odom_array)[:-1]

    def _create_indices(self):
        start_idx = int(self.cfg.start_time * 60)
        gap = round(self.cfg.snippet_gap * 60)
        return [
            i for i in range(start_idx, self.num_gaze, gap)
            if (i + self.input_length) < len(self.gaze_timestamps)
        ]

    def _get_crop_shape(self):
        # Can be overridden by child for MP4
        return (1408, 1408)

    def _init_rgb_handler(self):
        # Override in subclass
        return None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __del__(self):
        # override in child if needed
        pass

class RITWDataHandlerVRS(BaseRITWDataHandler):
    def __init__(self, cfg: DictConfig, vrs_file: str):
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
        super().__init__(cfg, vrs_file, is_vrs=True)
        self.provider = data_provider.create_vrs_data_provider(str(self.file_path)) if self.cfg.use_rgb else None
        self.stream_id = StreamId("214-1")
        self.gaze_xy = np.array(self.gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())

    def __getitem__(self, idx):
        i = self.indices[idx]
        snippet = {
            'gaze': self.gaze_sequence[i] if self.cfg.use_gaze else torch.zeros((self.input_length, 3), dtype=torch.float32),
            'odom': self.odometry_sequence[i] if self.cfg.use_imu else torch.zeros((self.input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        }
        if self.cfg.use_rgb and self.provider is not None and self.gaze_xy is not None:
            gaze_idx = i + self.input_length
            time_ns = self.gaze_timestamps[gaze_idx] * 1000  # μs to ns
            im_data = self.provider.get_image_data_by_time_ns(
                self.stream_id,
                time_ns,
                TimeDomain.DEVICE_TIME,         # <--- FIXED: use enum
                TimeQueryOptions.CLOSEST        # <--- FIXED: use enum
            )[0]
            im = im_data.to_numpy_array()
            im = cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)
            x_center = 1408 - np.clip(int(self.gaze_xy[gaze_idx, 0]), self.crop_size // 2, 1408 - self.crop_size // 2)
            y_center = np.clip(int(self.gaze_xy[gaze_idx, 1]), self.crop_size // 2, 1408 - self.crop_size // 2)
            gaze_crop = im[
                y_center - self.crop_size // 2 : y_center + self.crop_size // 2,
                x_center - self.crop_size // 2 : x_center + self.crop_size // 2
            ]
            snippet['rgb'] = (torch.Tensor(gaze_crop) / 255.).permute(2, 0, 1)
        timestamp = self.gaze_timestamps[i + self.input_length]
        return snippet, timestamp


class RITWDataHandlerMP4(BaseRITWDataHandler):
    def __init__(self, cfg: DictConfig, mp4_file: str):
        from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4
        super().__init__(cfg, mp4_file, is_vrs=False)
        self.frame_timestamps_ns = get_timestamp_from_mp4(str(self.file_path))
        self.cap = cv2.VideoCapture(str(self.file_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open MP4 file: {mp4_file}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Determine shape
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame0 = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read first frame from {mp4_file}")
        frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)
        self.rgb_height, self.rgb_width = frame0.shape[1], frame0.shape[0]  # w, h

    def get_frame_by_time_ns(self, target_ns):
        idx = int(np.argmin(np.abs(self.frame_timestamps_ns - target_ns)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {self.file_path}")
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    def __getitem__(self, idx):
        i = self.indices[idx]
        snippet = {
            'gaze': self.gaze_sequence[i] if self.cfg.use_gaze else torch.zeros((self.input_length, 3), dtype=torch.float32),
            'odom': self.odometry_sequence[i] if self.cfg.use_imu else torch.zeros((self.input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        }
        if self.cfg.use_rgb:
            gaze_idx = i + self.input_length
            time_ns = self.gaze_timestamps[gaze_idx] * 1000
            frame = self.get_frame_by_time_ns(time_ns)
            im = np.rot90(frame, k=-3)
            x_center = 1408 - np.clip(int(self.gaze_xy[gaze_idx, 0]), self.crop_size // 2, 1408 - self.crop_size // 2)
            y_center = np.clip(int(self.gaze_xy[gaze_idx, 1]), self.crop_size // 2, 1408 - self.crop_size // 2)
            gaze_crop = im[y_center - self.crop_size // 2:y_center + self.crop_size // 2, x_center - self.crop_size // 2:x_center + self.crop_size // 2]
            snippet['rgb'] = (torch.from_numpy(gaze_crop.copy()) / 255.).permute(2, 0, 1)
        timestamp = self.gaze_timestamps[i + self.input_length]
        return snippet, timestamp

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

# Example usage:
if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("your_predict.yaml", "r"))
    cfg['model_name'] = 'v1_default'
    cfg['use_rgb'] = True
    cfg['use_gaze'] = True
    cfg['use_imu'] = True
    cfg = DictConfig(cfg)
    # Use whichever handler fits your file type:
    dataset = RITWDataHandlerMP4(cfg, "/path/to/your.mp4")
    # dataset = RITWDataHandlerVRS(cfg, "/path/to/your.vrs")
