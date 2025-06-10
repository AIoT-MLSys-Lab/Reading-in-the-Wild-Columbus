import pandas as pd
from huggingface_hub import snapshot_download

def main():
    local_path = snapshot_download(
        "OSU-AIoT-MLSys-Lab/Reading-in-the-Wild-Columbus",
        repo_type='dataset',
        local_dir="../dataset",
    )

if __name__ == "__main__":
    main()