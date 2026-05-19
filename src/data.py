"""Dataset download, loading and temporal train/val/test split.

The benchmark targets ETTh1 (Electricity Transformer Temperature, hourly),
a standard time-series benchmark introduced by the Informer paper. ~17K
hourly rows, target column `OT` (oil temperature in Celsius).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import certifi

from .config import CONFIG, DATE_COLUMN, ETTH1_FILENAME, ETTH1_URL, RAW_DATA_DIR, TARGET_COLUMN


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


@dataclass
class TimeSeriesSplit:
    """Three temporally-ordered slices of the same series."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def describe(self) -> str:
        def _range(df: pd.DataFrame) -> str:
            if df.empty:
                return "(empty)"
            return f"{df.index.min()} -> {df.index.max()} ({len(df):,} rows)"

        return (
            f"train: {_range(self.train)}\n"
            f"val  : {_range(self.val)}\n"
            f"test : {_range(self.test)}"
        )


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = CONFIG.split.train_frac,
    val_frac: float = CONFIG.split.val_frac,
) -> TimeSeriesSplit:
    """Split the series chronologically.

    Random shuffling would leak future information into the training set, so we
    cut along the time axis instead.
    """

    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("Fractions must lie in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must leave room for a test set.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return TimeSeriesSplit(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
    )
