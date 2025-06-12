import os
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
# Your Hugging Face username. The script will try to get it automatically.
# If it fails, you can set it manually, e.g., "my-hf-username"
HF_USERNAME = None 
# The name of the repository you want to create on Hugging Face
REPO_NAME = "TinyVLA-droid_diffusion_metaworld"
# The local directory containing the model files you want to upload
LOCAL_MODEL_DIR = "checkpoints/TinyVLA-droid_diffusion_metaworld"
# ---------------------

api = HfApi()

if HF_USERNAME is None:
    try:
        HF_USERNAME = api.whoami()["name"]
        print(f"Successfully retrieved Hugging Face username: {HF_USERNAME}")
    except Exception as e:
        print(f"Could not automatically retrieve Hugging Face username: {e}")
        print("Please set the HF_USERNAME variable manually in this script.")
        exit()

# Form the full repository ID
repo_id = f"{HF_USERNAME}/{REPO_NAME}"

# Create the repository on the Hugging Face Hub
# The `exist_ok=True` flag prevents an error if the repo already exists.
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repository '{repo_id}' created or already exists on the Hub.")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit()

# Upload the contents of the local directory to the Hub
try:
    print(f"Uploading files from '{LOCAL_MODEL_DIR}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=LOCAL_MODEL_DIR,
        repo_id=repo_id,
        repo_type="model",
    )
    print("Upload complete!")
    print(f"Check out your model at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"An error occurred during upload: {e}") 