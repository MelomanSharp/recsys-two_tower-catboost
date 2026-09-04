# scripts/download_data.py
import os
import shutil
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
REQUIRED_FILES = [
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
    "sample_submission.csv",
]


def download_competition_data(competition_name: str, destination: Path) -> None:
    """
    Download Kaggle competition files through kagglehub and copy them to destination.
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("The kagglehub package is not installed. Run: pip install kagglehub")

    print(f"⏳ Downloading data for competition '{competition_name}'...")
    try:
        # competition_download returns a path to the extracted files.
        downloaded_path = kagglehub.competition_download(competition_name)
    except Exception as e:
        raise RuntimeError(f"kagglehub download failed: {e}")

    # Create the destination directory if needed.
    destination.mkdir(parents=True, exist_ok=True)

    # Copy all files from the temporary directory to destination.
    for file_name in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, file_name)
        dst = destination / file_name
        shutil.copy2(src, dst)  # copy2 preserves metadata.
        print(f"   Copied {file_name}")

    print(f"✅ All files copied to {destination}")


def main():
    # Exit when all required files are already present.
    if all((RAW_DATA_DIR / f).exists() for f in REQUIRED_FILES):
        print("✅ All CSV files already exist in data/raw/. Skipping download.")
        return

    # Otherwise, download through the Kaggle API.
    competition = "h-and-m-personalized-fashion-recommendations"
    try:
        download_competition_data(competition, RAW_DATA_DIR)
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        print("Make sure you have internet access and configured Kaggle credentials.")
        print("Setup instructions: https://www.kaggle.com/docs/api#authentication")
        raise

    # Verify that all files were downloaded.
    missing = [f for f in REQUIRED_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print(f"⚠️ Files missing after download: {missing}")
    else:
        print("✅ Download completed successfully.")


if __name__ == "__main__":
    main()