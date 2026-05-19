"""Download the ETTh1 dataset into ``data/raw/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script: ``python scripts/download_data.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import download_etth1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = parser.parse_args()
    path = download_etth1(force=args.force)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
