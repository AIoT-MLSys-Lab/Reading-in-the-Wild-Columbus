import os
import logging
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Import the inference function from your single-file prediction module.
# Adjust the module name as needed.
from predict_single_file import inference
from src.predict_folder import process_all_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_single_file(cfg: DictConfig):
    """
    Process a single VRS file using the inference function.
    """
    original_cwd = Path(get_original_cwd())
    # Build full input path from root_dir and input_filename.
    input_path = original_cwd / cfg.root_dir / cfg.input_filename
    if not input_path.exists():
        logging.error("Input VRS file not found: %s", input_path)
        return

    logging.info("Processing single file: %s", input_path)
    # Create a copy of the config and set the input_filename.
    file_cfg = cfg.copy()
    file_cfg.input_filename = input_path.name

    # Run inference.
    timestamps, predictions = inference(file_cfg)

    vid_uid = input_path.stem
    output_dir = Path(cfg.output_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{vid_uid}.csv"

    df = pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
    df.to_csv(str(output_file), index=False)
    logging.info("Predictions for %s saved to %s", vid_uid, output_file)


def process_file(file: Path, base_cfg: DictConfig):
    """
    Process a single VRS file for batch processing.
    """
    try:
        logging.info("Processing file: %s", file)
        # Create a copy of the config and update the input_filename for this file.
        file_cfg = base_cfg.copy()
        file_cfg.input_filename = file.name

        # Run inference using the imported function.
        timestamps, predictions = inference(file_cfg)

        vid_uid = file.stem
        output_dir = Path(base_cfg.output_save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{vid_uid}.csv"

        df = pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
        df.to_csv(str(output_file), index=False)
        logging.info("Predictions for %s saved to %s", vid_uid, output_file)
        return vid_uid, True
    except Exception as e:
        logging.error("Error processing file %s: %s", file, e)
        return str(file), False


def process_folder(cfg: DictConfig):
    """
    Process all VRS files in the specified root directory in parallel.
    """
    original_cwd = Path(get_original_cwd())
    root_dir = original_cwd / cfg.root_dir
    vrs_files = list(root_dir.glob("*.vrs"))
    if not vrs_files:
        logging.error("No VRS files found in %s", root_dir)
        return

    logging.info("Found %d VRS files in %s", len(vrs_files), root_dir)
    args = [(file, cfg, original_cwd) for file in vrs_files]

    with Pool(processes=cfg.num_workers) as pool:
        results = pool.starmap(process_file, args)

    processed = sum(1 for _, success in results if success)
    logging.info("Folder processing complete: %d/%d files successfully processed.", processed, len(vrs_files))


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    if 'input_filename' in cfg:
        process_single_file(cfg)
    else:
        process_all_files(cfg)


if __name__ == "__main__":
    main()
