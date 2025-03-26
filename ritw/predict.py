import logging
import time
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, set_start_method

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

from ritw.constants import MODALITY_MAPPING
from ritw.predict_single_file import inference
from ritw.utils import get_vrs_file_list

# Use file_system sharing to avoid mmap issues.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

EXPECTED_MODALITIES = lambda cfg: len(cfg.modalities)


def process_data(file: Path, base_cfg: DictConfig, model: str, orig_cwd: Path):
    """
    For a given file and model, run inference sequentially for all modalities,
    merge the results, and write to a CSV.
    """
    logging.info("Processing file %s with model %s", file, model)
    merged_results = {}
    # Process modalities sequentially
    for modality in base_cfg.modalities:
        try:
            cfg = base_cfg.copy()
            with open_dict(cfg):
                cfg['input_filename'] = file.name
                cfg['original_cwd'] = str(orig_cwd)
            cfg.model_name = model
            mod_flags = MODALITY_MAPPING.get(modality)
            if mod_flags is None:
                raise ValueError(f"Unknown modality combination: {modality}")
            with open_dict(cfg):
                cfg['use_gaze'] = mod_flags["use_gaze"]
                cfg['use_imu'] = mod_flags["use_imu"]
                cfg['use_rgb'] = mod_flags["use_rgb"]
            timestamps, predictions = inference(cfg)
            if timestamps is None or predictions is None:
                logging.warning("No results for modality %s for file %s, model %s", modality, file, model)
                continue
            logging.info("Completed modality %s for file %s with model %s", modality, file.stem, model)
            merged_results[modality] = (timestamps, predictions)
        except Exception as e:
            logging.error("Error processing modality %s for file %s, model %s: %s", modality, file, model, e)
    # Check that we got results for all modalities (or at least some)
    if not merged_results:
        logging.error("No valid modality results for file %s, model %s", file, model)
        return file.stem

    # Use timestamps from one modality (if they differ, a warning would have been issued)
    merged_data = {}
    for modality, (ts, preds) in merged_results.items():
        if "timestamp" not in merged_data:
            merged_data["timestamp"] = ts
        else:
            if merged_data["timestamp"] != ts:
                logging.warning("Timestamp mismatch for file %s, model %s on modality %s", file.stem, model, modality)
        merged_data[modality] = preds

    df = pd.DataFrame(merged_data)
    output_dir = Path(orig_cwd) / Path(base_cfg.output_save_path) / model
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file.stem}_{model}.csv"
    df.to_csv(str(output_file), index=False)
    logging.info("Merged predictions for file %s with model %s saved to %s", file.stem, model, output_file)
    return file.stem


def get_file_model_tasks(cfg: DictConfig, orig_cwd: Path):
    """
    Build a list of tasks (file, model) based on mode.
    """
    if cfg.mode.lower() == "single":
        file_path = orig_cwd / cfg.root_dir / cfg.input_filename
        files = [file_path] if file_path.exists() else []
    else:
        root_dir = orig_cwd / cfg.root_dir
        files = list(root_dir.glob("*.vrs"))
    tasks = []
    for file in files:
        for model in cfg.model_name:
            tasks.append((file, cfg, model, orig_cwd))
    return tasks


@hydra.main(config_path="../config", config_name="predict", version_base="1.2")
def main(cfg: DictConfig):
    orig_cwd = Path(get_original_cwd())
    logging.info("Original working directory: %s", orig_cwd)
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    tasks = get_file_model_tasks(cfg, orig_cwd)
    if not tasks:
        logging.error("No files to process.")
        return
    logging.info("Processing %d file-model tasks.", len(tasks))
    with Pool(processes=cfg.num_workers) as pool:
        pool.starmap(process_data, tasks)
    logging.info("All tasks completed.")

if __name__ == "__main__":
    main()
