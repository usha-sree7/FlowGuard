"""Download (from the authors' GitHub repositories), verify, standardise and validate a dataset.

    python scripts/import_dataset.py bengalas
    python scripts/import_dataset.py malwathu
    python scripts/import_dataset.py bengalas --no-download   # files already placed in data/raw/bengalas/
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floodguard.datasets import DatasetError, import_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["bengalas", "malwathu", "cwc_chindnar"])
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    try:
        rep = import_dataset(a.dataset, download_files=not a.no_download, progress=lambda m: print("  ..", m))
    except DatasetError as e:
        print(f"IMPORT FAILED: {e}")
        sys.exit(1)
    keys = ["rows", "valid_rows", "first", "last", "gaps", "largest_gap_hours", "duplicate_timestamps",
            "missing_values", "rise_rate_available"]
    print(json.dumps({k: rep[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
