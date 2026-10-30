"""
Upload the fine-tuned expert checkpoint to Hugging Face Hub.
"""

from pathlib import Path
from huggingface_hub import HfApi, login


def main():
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_dir = repo_root / "models" / "checkpoints" / "qwen2.5-0.5b-expert"

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return

    print("=== Publish Fine-Tuned Model to Hugging Face ===")
    token = input("Enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Error: Hugging Face token is required.")
        return

    print("Authenticating...")
    login(token=token)
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("username")
    repo_id = f"{username}/qwen2.5-0.5b-expert"

    print(f"Creating repository https://huggingface.co/{repo_id} (if not already exists)...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading files from {checkpoint_dir}...")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"\n✓ Published successfully to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
