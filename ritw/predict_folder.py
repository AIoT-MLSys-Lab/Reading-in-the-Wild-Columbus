import os
import logging
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

# Import the inference function from your single-file prediction module.
from predict_single_file import inference

# Configure logging (if not already configured in the imported module)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_file(file: Path, base_cfg: DictConfig):
    """
    Process a single VRS file using the single-file inference function.
    Returns the video UID and a success flag.
    """
    try:
        logging.info("Processing file: %s", file)
        # Create a copy of the config for this file and update the input_filename field.
        file_cfg = base_cfg.copy()
        with open_dict(file_cfg):
            file_cfg['input_filename'] = file.name

        # Run inference for this file.
        timestamps, predictions = inference(file_cfg)

        # Use the file stem as the video UID.
        vid_uid = file.stem

        # Save predictions to a CSV file.
        output_dir = Path(base_cfg.output_save_path) / file_cfg.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{vid_uid}.csv"
        df = pd.DataFrame({'timestamp': timestamps, 'prediction': predictions})
        df.to_csv(str(output_file), index=False)
        logging.info("Predictions for %s saved to %s", vid_uid, output_file)
        return vid_uid, True
    except Exception as e:
        logging.error("Error processing file %s: %s", file, e)
        return str(file), False


def process_all_files(cfg: DictConfig):
    """
    Locate all VRS files in the specified root directory and process them in parallel.
    """
    original_cwd = Path(get_original_cwd())
    root_dir = original_cwd / cfg.root_dir
    vrs_files = list(root_dir.glob("*.vrs"))
    if not vrs_files:
        logging.error("No VRS files found in %s", root_dir)
        return

    logging.info("Found %d VRS files in %s", len(vrs_files), root_dir)

    # Prepare arguments for multiprocessing.
    args = [(file, cfg) for file in vrs_files]

    with Pool(processes=cfg.num_workers) as pool:
        results = pool.starmap(process_file, args)

    processed = sum(1 for _, success in results if success)
    logging.info("Processing complete: %d/%d files successfully processed.", processed, len(vrs_files))


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logging.info("Folder processing with configuration:\n%s", OmegaConf.to_yaml(cfg))
    process_all_files(cfg)


if __name__ == "__main__":
    main()
