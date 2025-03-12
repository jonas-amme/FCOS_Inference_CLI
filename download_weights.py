import os
import requests
from tqdm import tqdm 

# Configurations
CHECKPOINT_DIR = "checkpoints"
RELEASE_TAG = "v1.0.0"
REPO_OWNER = "jonas-amme"
REPO_NAME = "FCOS_Inference_CLI"

# Model files to download 
MODEL_FILES = [
    "FCOS_18.ckpt",
    "FCOS_x50.ckpt",
    "FCOS_x101.ckpt"
]

def download_file(url, filepath):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))

        # Create progress bar 
        pbar = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(filepath)}"
        )

        # Download the file 
        with open(filepath, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        pbar.close()
        return True
    return False


def main():
    # Create checkpoint directory if it does not exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Download each model file
    for model_file in MODEL_FILES:
        url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{model_file}"
        filepath = os.path.join(CHECKPOINT_DIR, model_file)

        print(f"\nDownlading {model_file} ...")
        if download_file(url, filepath):
            print(f"Successfully downloaded {model_file}.")
        else:
            print(f"Failed to download {model_file}.")


if __name__ == "__main__":
    main()
