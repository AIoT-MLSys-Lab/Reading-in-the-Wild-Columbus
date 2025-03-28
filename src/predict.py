import logging
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, set_start_method

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

# Import the inference function from your single-file prediction module.
from predict_single_file import inference

# Set the multiprocessing start method early (use "spawn" to avoid fork issues).
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping from config modality nomenclature to flag settings.
MODALITY_MAPPING = {
    "gaze": {"use_gaze": True, "use_imu": False, "use_rgb": False},
    "imu": {"use_gaze": False, "use_imu": True, "use_rgb": False},
    "rgb": {"use_gaze": False, "use_imu": False, "use_rgb": True},
    "gaze+rgb": {"use_gaze": True, "use_imu": False, "use_rgb": True},
    "gaze+imu": {"use_gaze": True, "use_imu": True, "use_rgb": False},
    "imu+rgb": {"use_gaze": False, "use_imu": True, "use_rgb": True},
    "gaze+imu+rgb": {"use_gaze": True, "use_imu": True, "use_rgb": True},
}


def process_modality(file: Path, base_cfg: DictConfig, model: str, modality: str, orig_cwd: Path):
    """
    Process a single VRS file with a given model and modality combination.
    Returns: (file_stem, model, modality, timestamps, predictions)
    """
    try:
        logging.info("Processing %s for file: %s using model %s", modality, file, model)
        cfg = base_cfg.copy()
        # Set file and model info.
        with open_dict(cfg):
            cfg['input_filename'] = file.name
        cfg.model_name = model
        # Set original_cwd in case inference needs it.
        with open_dict(cfg):
            cfg['original_cwd'] = str(orig_cwd)
        # Update modality flags based on mapping.
        mod_flags = MODALITY_MAPPING.get(modality)
        if mod_flags is None:
            raise ValueError(f"Unknown modality combination: {modality}")
        with open_dict(cfg):
            cfg['use_gaze'] = mod_flags["use_gaze"]
            cfg['use_imu'] = mod_flags["use_imu"]
            cfg['use_rgb'] = mod_flags["use_rgb"]
        # Run inference.
        timestamps, predictions = inference(cfg)
        logging.info("Completed %s for file %s with model %s", modality, file.stem, model)
        return file.stem, model, modality, timestamps, predictions
    except Exception as e:
        logging.error("Error processing file %s, model %s, modality %s: %s", file, model, modality, e)
        return file.stem, model, modality, None, None


def merge_file_results(results):
    """
    Merge inference results from different modality and model combinations for a single file.
    Returns a dict keyed by (file_stem, model) with value DataFrame.
    Each DataFrame has columns: timestamp and one column per modality.
    """
    merged = {}
    for file_stem, model, modality, timestamps, predictions in results:
        if timestamps is None or predictions is None:
            continue
        key = (file_stem, model)
        if key not in merged:
            merged[key] = {"timestamp": timestamps}
        else:
            if merged[key]["timestamp"] != timestamps:
                logging.warning("Timestamp mismatch for %s and model %s on modality %s", file_stem, model, modality)
        # Use modality name as column (replace '+' with '_' if desired)
        col_name = modality
        merged[key][col_name] = predictions
    dfs = {}
    for key, data in merged.items():
        dfs[key] = pd.DataFrame(data)
    return dfs


def get_file_list(cfg: DictConfig, orig_cwd: Path):
    """
    Get list of VRS files based on mode. For single mode, return one file; for folder, all .vrs files.
    """
    if cfg.mode.lower() == "single":
        file_path = orig_cwd / cfg.root_dir / cfg.input_filename
        return [file_path] if file_path.exists() else []
    else:
        root_dir = orig_cwd / cfg.root_dir
        return list(root_dir.glob("*.vrs"))[:2]


def process_all(cfg: DictConfig, orig_cwd: Path):
    """
    Process files for all specified models and modalities in parallel.
    For each file, run inference for each modality combination and model.
    Merge the results and save a CSV for each (file, model) pair.
    """
    files = get_file_list(cfg, orig_cwd)
    if not files:
        logging.error("No VRS files found.")
        return

    models = cfg.model_name  # List of model names.
    modalities = cfg.modalities  # List of modality strings.
    logging.info("Processing %d files with models %s and modalities %s", len(files), models, modalities)

    tasks = []
    for file in files:
        for model in models:
            for modality in modalities:
                tasks.append((file, cfg, model, modality, orig_cwd))
    logging.info("Total tasks: %d", len(tasks))

    with Pool(processes=cfg.num_workers) as pool:
        results = pool.starmap(process_modality, tasks)

    merged_dfs = merge_file_results(results)
    for (file_stem, model), df in merged_dfs.items():
        output_dir = Path(orig_cwd).parent / Path(cfg.output_save_path) / model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_stem}_{model}.csv"
        df.to_csv(str(output_file), index=False)
        logging.info("Merged predictions for %s with model %s saved to %s", file_stem, model, output_file)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get the original working directory once, after Hydra is initialized.
    orig_cwd = Path(get_original_cwd())
    logging.info("Original working directory: %s", orig_cwd)
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    process_all(cfg, orig_cwd)


if __name__ == "__main__":
    main()
