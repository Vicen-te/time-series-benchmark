"""Dataset download, loading and temporal train/val/test split.

The benchmark targets ETTh1 (Electricity Transformer Temperature, hourly),
a standard time-series benchmark introduced by the Informer paper. ~17K
hourly rows, target column `OT` (oil temperature in Celsius).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import certifi

from .config import ETTH1_FILENAME, ETTH1_URL, RAW_DATA_DIR


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
