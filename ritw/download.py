import logging
import os
import pandas as pd
import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import list_repo_files


def download_metadata(repo_id, local_dir):
    """Download metadata.csv only."""
    return hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename='metadata.csv',
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

def filter_metadata(metadata_path, filter_dict):
    """Filter metadata.csv based on filter_dict."""
    df = pd.read_csv(metadata_path)
    for col, vals in filter_dict.items():
        df = df[df[col].isin(vals)]
    return df

def download_uid_files(repo_id, local_dir, uid):
    """Download all files for a given UID."""
    # Download calib/<uid>.pkl
    hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=f'calib/{uid}.pkl',
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    # Download mp4/<uid>.mp4
    hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=f'mp4/{uid}.mp4',
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    # Download all files in mps/mps_<uid>_vrs/
    # List files in that folder (using HF API)
    prefix = f"mps/mps_{uid}_vrs/"
    files = [f for f in list_repo_files(repo_id, repo_type='dataset') if f.startswith(prefix)]
    for fpath in files:
        hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=fpath,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

@hydra.main(config_path="../config", config_name="download", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    repo_id = cfg.dataset_repo
    local_dir = cfg.local_dir
    filter_dict = dict(cfg.filter) if 'filter' in cfg else {}

    os.makedirs(local_dir, exist_ok=True)
    # Download metadata
    logging.info(f"Downloading metadata {repo_id}...")
    metadata_path = download_metadata(repo_id, local_dir)
    # Filter
    filtered_df = filter_metadata(metadata_path, filter_dict)
    print(f"Found {len(filtered_df)} matches, total duration: {filtered_df['Duration'].sum()/60:.2f} minutes")
    proceed = input(f"\nProceed with download? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Aborted.")
        return
    # Download related files
    for uid in filtered_df['UID']:
        logging.info(f"Downloading files for UID: {uid}")
        download_uid_files(repo_id, local_dir, uid)

if __name__ == "__main__":
    main()

