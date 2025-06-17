import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ritw.data_handler import RITWDataHandlerMP4
from ritw.data_handler import RITWDataHandlerVRS

# Import your own modules and constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inference(cfg: DictConfig):
    logging.info("Using modalities - rgb: %s, imu: %s, gaze: %s", cfg.use_rgb, cfg.use_imu, cfg.use_gaze)

    num_classes = 2
    model_name = cfg.model_name  # e.g., 'v1_default'
    if model_name == 'v1_mode':
        num_classes = 7  # not reading, out loud, normal, scan, walking, write/type, skim
    if model_name == 'v1_medium':
        num_classes = 4  # not reading, print, digital, object

    # Create output folder if needed.
    output_save_path = cfg.output_save_path
    os.makedirs(output_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'device' in cfg:
        device = torch.device(cfg.device)

    # Load TorchScript model from the ../models directory.
    original_cwd = Path(cfg.original_cwd)
    model_dir = original_cwd / Path("models")
    model_path = model_dir / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.jit.load(str(model_path))
    model = model.to(device)

    # Build full input path from root_dir and input_filename.
    input_path = Path(cfg.root_dir) / cfg.input_filename

    # Create the dataset using the VRS file (all processing is done within the dataset).
    if 'file_type' in cfg and cfg.file_type == 'mp4':
        dataset = RITWDataHandlerMP4(cfg, str(input_path))
    else:
        dataset = RITWDataHandlerVRS(cfg, str(input_path))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference loop.
    stacked_predictions = []
    timestamps = []
    for batch, timestamp in data_loader:
        # Always pass tensors to the model—do not pass None.
        gaze_input = batch['gaze'].to(device)
        odom_input = batch['odom'].to(device)
        rgb_input = batch['rgb'].to(device)
        with torch.no_grad():
            pred = model(gaze_input, odom_input, rgb_input)
        stacked_predictions.append(pred)
        timestamps.append(timestamp.item())

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


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Display the configuration for verification.
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    # Run inference using the unchanged function interface.
    timestamps, predictions = inference(cfg)

    # Save predictions in a CSV file; the output file is named using the base name of the input file.
    os.makedirs(cfg.output_save_path, exist_ok=True)
    vid_uid = Path(cfg.input_filename).stem
    output_file = os.path.join(cfg.output_save_path, f"{vid_uid}.csv")
    df = pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
    df.to_csv(output_file, index=False)
    logging.info("Predictions saved to %s", output_file)


if __name__ == "__main__":
    main()
