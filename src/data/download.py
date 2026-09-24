"""Download the LinkedIn Job Postings dataset from Kaggle into data/raw/.

Credentials are read by the Kaggle client from ~/.kaggle/kaggle.json.
They are never stored in this repo.

Usage:
    python -m src.data.download          # skip if already downloaded
    python -m src.data.download --force  # re-download
"""

import argparse
import sys
from pathlib import Path

DATASET = "arshkon/linkedin-job-postings"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MARKER_FILE = RAW_DIR / "postings.csv"


def download(force: bool = False) -> Path:
    if MARKER_FILE.exists() and not force:
        print(f"Already downloaded: {MARKER_FILE}")
        return RAW_DIR

    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit(
            "Missing ~/.kaggle/kaggle.json. Create an API token at "
            "https://www.kaggle.com/settings and place the file there."
        )

    # Imported here because the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True, quiet=False)
    print("Done.")
    return RAW_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download")
    download(force=parser.parse_args().force)
