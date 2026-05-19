"""Dataset download, loading and temporal train/val/test split.

The benchmark targets ETTh1 (Electricity Transformer Temperature, hourly),
a standard time-series benchmark introduced by the Informer paper. ~17K
hourly rows, target column `OT` (oil temperature in Celsius).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import certifi

from .config import DATE_COLUMN, ETTH1_FILENAME, ETTH1_URL, RAW_DATA_DIR, TARGET_COLUMN


def download_etth1(force: bool = False, dest_dir: Optional[Path] = None) -> Path:
    """Download ETTh1.csv into ``data/raw/`` if not already present."""

    dest_dir = dest_dir or RAW_DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ETTH1_FILENAME

    if dest.exists() and not force:
        return dest

    # certifi bundles up-to-date root certificates; using it avoids the
    # SSL verification failures Windows-bundled Python sometimes produces.
    response = requests.get(ETTH1_URL, timeout=60, verify=certifi.where())
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_etth1(
    target: str = TARGET_COLUMN,
    csv_path: Optional[Path] = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load ETTh1 as a DataFrame indexed by hourly timestamp.

    Returns a frame with all original columns plus a DatetimeIndex.
    """

    csv_path = csv_path or (RAW_DATA_DIR / ETTH1_FILENAME)
    if not csv_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"{csv_path} not found. Run `python scripts/download_data.py` first."
            )
        csv_path = download_etth1()

    df = pd.read_csv(csv_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' missing. Available: {list(df.columns)}")

    return df
