#!/usr/bin/env python
import os
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.utils import get_original_cwd

from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Hard-coded modality map (for display purposes)
MODALITY_MAP = {
    'gaze': {'Gaze': True, 'RGB': False, 'IMU': False},
    'rgb': {'Gaze': False, 'RGB': True, 'IMU': False},
    'imu': {'Gaze': False, 'RGB': False, 'IMU': True},
    'gaze+imu': {'Gaze': True, 'RGB': False, 'IMU': True},
    'imu+rgb': {'Gaze': False, 'RGB': True, 'IMU': True},
    'gaze+rgb': {'Gaze': True, 'RGB': True, 'IMU': False},
    'gaze+imu+rgb': {'Gaze': True, 'RGB': True, 'IMU': True},
}


def map_bool_to_str(modality_mapping):
    """Convert a boolean modality mapping into a simple string representation."""
    return {k: '[x]' if v else '[ ]' for k, v in modality_mapping.items()}


def classification_metrics(y_true, y_prob, target_recall=0.9, metrics_set=None):
    """
    Computes classification metrics: F1, Accuracy, Precision/Threshold at a target recall, and AUC.
    Returns a dictionary with the requested metrics.
    """
    y_pred = y_prob > 0.5
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    roc_auc = auc(recalls, precisions)
    thresholds = np.concatenate(([thresholds[0]], thresholds))
    precision_at_req = np.interp(target_recall, recalls[::-1], precisions[::-1])
    threshold_at_req = np.interp(target_recall, recalls[::-1], thresholds[::-1])
    f1_at_r = f1_score(y_true, y_prob > threshold_at_req)
    acc_at_r = accuracy_score(y_true, y_prob > threshold_at_req)
    res = {
        'F1': f1,
        'Acc': acc,
        'P@R=0.9': precision_at_req,
        'T@R=0.9': threshold_at_req,
        'Acc@R=0.9': acc_at_r,
        'F1@R=0.9': f1_at_r,
        'AUC': roc_auc
    }
    if metrics_set is not None:
        res = {k: res[k] for k in res if k in metrics_set}
    res = {k: round(v, 4) for k, v in res.items()}
    return res


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply filtering to the DataFrame based on the filters dictionary.
    Each key in filters is a column name, and its value is a list of allowed values.
    """
    for col, allowed in filters.items():
        if col in df.columns:
            df = df[df[col].isin(allowed)]
    return df


@hydra.main(config_path="../config", config_name="evaluate", version_base="1.2")
def main(cfg: DictConfig):
    # Get the original working directory (before Hydra changes it)
    orig_cwd = Path(get_original_cwd())
    logging.info("Original working directory: %s", orig_cwd)
    logging.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Load metadata from file.
    metadata_path = Path(cfg.metadata_file)
    if not metadata_path.exists():
        logging.error("Metadata file not found: %s", metadata_path)
        return
    metadata = pd.read_csv(str(metadata_path))

    # Apply filters from config.
    if "filters" in cfg:
        metadata = apply_filters(metadata, cfg.filters)
        logging.info("Applied filters; %d rows remain.", len(metadata))
    else:
        logging.info("No filters specified; using all metadata rows.")

    # Find processed CSV files in the result directory.
    result_dir = cfg.result_dir  # e.g. "output/v1_default"
    csv_file_paths = []
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_file_paths.append(os.path.join(root, file))
    logging.info("Found %d CSV files in %s", len(csv_file_paths), result_dir)

    def find_if_model_processed(x):
        present = False
        output_file_name = x['filename']
        for csv_file in csv_file_paths:
            if output_file_name in csv_file:
                present = True
        return present

    csv_set = set()

    def find_corresponding_csv_file(x):
        filename = None
        output_data = x['filename']
        for csv_file in csv_file_paths:
            if output_data in csv_file and csv_file not in csv_set:
                filename = csv_file
                csv_set.add(csv_file)
                break
        return filename

    metadata_valid = metadata[metadata.apply(find_if_model_processed, axis=1)]
    metadata_valid.loc[:, 'csv_file'] = metadata_valid.apply(find_corresponding_csv_file, axis=1)

    # For each valid row, compute percentage predictions.
    def percentage_one_zero(row):
        data = pd.read_csv(row['csv_file'], index_col='timestamp')
        percentage = data.sum() / len(data)
        updated_row = row.copy()
        for col in data.columns:
            updated_row[f"read_prediction_with_{col}"] = percentage[col]
        return updated_row

    metadata_valid = metadata_valid.apply(percentage_one_zero, axis=1)

    # Process the metadata to compute overall metrics.
    all_data = []
    # Assume metadata has these columns: ActivityType, Medium, ShortTextOrPara, csv_file, ContainsNonText, Platform, ExtraTags, filename.
    roc_data = metadata_valid[
        ['ActivityType', 'Medium', 'ShortTextOrPara', 'csv_file', 'ContainsNonText', 'Platform', 'ExtraTags',
         'filename']]

    # Define helper: convert ActivityType to binary label.
    def activity_to_binary(x):
        if x in ['reading_positive', 'reading_searching']:
            return 1
        return 0

    roc_data['binary_act'] = roc_data['ActivityType'].apply(activity_to_binary)

    for _, row in tqdm(roc_data.iterrows(), total=len(roc_data)):
        cat = row['binary_act']
        fname = row['csv_file']
        medium = row['Medium']
        content = row['ShortTextOrPara']
        df = pd.read_csv(fname, index_col=None)
        if len(df) == 0:
            logging.warning("Empty CSV for %s", fname)
            continue
        for k, v in row.items():
            df[k] = v
        df['act'] = cat
        all_data.append(df)

    if not all_data:
        logging.error("No valid results found.")
        return
    df_all = pd.concat(all_data)

    # Now compute metrics per modality.
    res = {}
    for col in cfg.modalities:
        if col not in df_all.columns:
            logging.warning("Modality %s not found in data columns", col)
            continue
        modality_str = map_bool_to_str(MODALITY_MAP.get(col, {}))
        df_num = df_all[['act', col]].dropna()
        res[col] = modality_str | classification_metrics(
            df_num['act'], df_num[col],
            target_recall=cfg.target_recall,
            metrics_set=set(cfg.metrics)
        )

    res_df = pd.DataFrame(res).T.reset_index(drop=True)
    print(res_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
