import os
import logging
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

import hydra
from omegaconf import DictConfig, OmegaConf

# Import your own modules and constants
from utils import project_gaze_vrs, create_sampled_array

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inference(cfg: DictConfig):
    logging.info("Using modalities - rgb: %s, imu: %s, gaze: %s", cfg.use_rgb, cfg.use_imu, cfg.use_gaze)

    # parameters
    input_hz = 60
    input_sec = 2
    crop_size = 64
    num_classes = 2

    model_name = cfg.model_name  # e.g., 'v1_default'
    if model_name == 'v1_15Hz':
        input_hz = 15
    if model_name == 'v1_1s':
        input_sec = 1
    if model_name == 'v1_mode':
        num_classes = 7  # not reading, out loud, normal, scan, walking, write/type, skim
    if model_name == 'v1_medium':
        num_classes = 4  # not reading, print, digital, object
    if model_name == 'v1_large':
        crop_size = 128

    input_length = input_hz * input_sec

    # saving output folder
    output_save_path = cfg.output_save_path
    os.makedirs(output_save_path, exist_ok=True)

    # Load TorchScript model from the ../models directory
    original_cwd = Path(get_original_cwd())
    model_dir = original_cwd / Path("../models")
    model_path = model_dir / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.jit.load(str(model_path))
    model = model.cuda()

    # Build full input path from root_dir and input_filename
    input_path = Path(cfg.root_dir) / cfg.input_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input VRS file not found: {input_path}")
    vrs_path = str(input_path)

    # Extract vid_uid from the input filename
    vid_uid = input_path.stem

    # Determine gaze file path; check for personalized, then general
    mps_folder = input_path.parent / f"mps_{vid_uid}_vrs" / "eye_gaze"
    gaze_path = mps_folder / "personalized_eye_gaze.csv"
    if not gaze_path.exists():
        gaze_path = mps_folder / "general_eye_gaze.csv"
    if not gaze_path.exists():
        raise FileNotFoundError(f"Gaze CSV file not found in {mps_folder}")

    # Process gaze data
    gaze = project_gaze_vrs(str(gaze_path), vrs_path=vrs_path)

    # Process IMU data if requested
    if cfg.use_imu:
        odom_path = input_path.parent / f"mps_{vid_uid}_vrs" / "slam" / "open_loop_trajectory.csv"
        if not odom_path.exists():
            raise FileNotFoundError(f"Odometry CSV file not found: {odom_path}")
        odometry = pd.read_csv(odom_path, engine='python')
        odometry = odometry[["device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry",
                             "device_linear_velocity_z_odometry", "angular_velocity_x_device",
                             "device_linear_velocity_y_odometry", "angular_velocity_z_device"]]
        indices = np.linspace(0, len(odometry) - 1, len(gaze), dtype=int)  # resample to 60Hz
        odometry = odometry.iloc[indices]

    # If using RGB, initialize the data provider
    if cfg.use_rgb:
        provider = data_provider.create_vrs_data_provider(vrs_path)
        deliver_option = provider.get_default_deliver_queued_options()
        deliver_option.deactivate_stream_all()
        deliver_option.activate_stream(StreamId("214-1"))

    # Input processing: create 2s long snippets [T-2, T] for prediction at time T
    gaze_sequence = gaze[['transformed_gaze_x', 'transformed_gaze_y', 'transformed_gaze_z']].ffill()
    gaze_sequence = create_sampled_array(gaze_sequence, num_samples=input_length + 1, stride=60 // input_hz)
    gaze_sequence = torch.Tensor(np.diff(gaze_sequence, axis=1) * input_hz)
    num_gaze = gaze_sequence.size(0)
    gaze_timestamps = gaze['tracking_timestamp_us'].tolist()

    if cfg.use_imu:
        odometry_sequence = create_sampled_array(odometry, num_samples=input_length, stride=60 // input_hz)
        odometry_sequence = torch.Tensor(odometry_sequence)[:-1]

    if cfg.use_rgb:
        gaze_xy = np.array(gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())

    test_set = []
    timestamps = []
    # Process snippets starting from a given start time and using a snippet gap (in seconds)
    for i in range(int(cfg.start_time * 60), num_gaze, round(cfg.snippet_gap * 60)):
        snippet = {
            'gaze': torch.zeros((input_length, 3), dtype=torch.float32),
            'odom': torch.zeros((input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, crop_size, crop_size), dtype=torch.float32)
        }
        if cfg.use_gaze:
            snippet['gaze'] = gaze_sequence[i]
        if cfg.use_imu:
            snippet['odom'] = odometry_sequence[i]
        if cfg.use_rgb:
            gaze_idx = i + input_length  # use RGB at the end of gaze sequence
            time_ns = gaze_timestamps[gaze_idx] * 1000  # convert from microseconds to nanoseconds
            im_data = provider.get_image_data_by_time_ns(StreamId("214-1"), time_ns, TimeDomain.DEVICE_TIME,
                                                         TimeQueryOptions.CLOSEST)[0]
            im = im_data.to_numpy_array()
            # Rotate and convert image colors
            im = cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)
            x_ = 1408 - np.clip(int(gaze_xy[gaze_idx, 0]), crop_size // 2, 1408 - crop_size // 2)
            y_ = np.clip(int(gaze_xy[gaze_idx, 1]), crop_size // 2, 1408 - crop_size // 2)
            gaze_crop = im[y_ - crop_size // 2:y_ + crop_size // 2, x_ - crop_size // 2:x_ + crop_size // 2]
            snippet['rgb'] = (torch.Tensor(gaze_crop) / 255.).permute(2, 0, 1)
        test_set.append(snippet)
        timestamps.append(gaze_timestamps[i + input_length])

    data_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Inference loop
    stacked_predictions = []
    for batch in data_loader:
        with torch.no_grad():
            pred = model(batch['gaze'].cuda(), batch['odom'].cuda(), batch['rgb'].cuda())
        stacked_predictions.append(pred)
    stacked_predictions = torch.cat(stacked_predictions, dim=0)

    if num_classes > 2:
        pred = torch.argmax(stacked_predictions, -1)
    else:
        stacked_predictions = F.softmax(stacked_predictions, -1)
        pred = stacked_predictions[:, 1].cpu().numpy()

    if num_classes <= 2:
        read_det = 100 * sum(pred > 0.5) / len(pred)
        logging.info("Read Detection Percentage: %.2f%%", read_det)

    return timestamps, pred.tolist()


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Display the configuration for verification
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Run inference on the single input file specified by the config
    timestamps, predictions = inference(cfg)

    # Save predictions in a CSV file; the output file is named using the base name of the input file
    os.makedirs(cfg.output_save_path, exist_ok=True)
    vid_uid = Path(cfg.input_filename).stem
    output_file = os.path.join(cfg.output_save_path, f"{vid_uid}.csv")
    df = pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
    df.to_csv(output_file, index=False)
    logging.info("Predictions saved to %s", output_file)


if __name__ == "__main__":
    main()
