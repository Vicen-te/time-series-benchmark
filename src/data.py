"""Dataset download, loading and temporal train/val/test split.

The benchmark targets ETTh1 (Electricity Transformer Temperature, hourly),
a standard time-series benchmark introduced by the Informer paper. ~17K
hourly rows, target column `OT` (oil temperature in Celsius).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import certifi

from .config import (
    CONFIG,
    DATE_COLUMN,
    ETTH1_FILENAME,
    ETTH1_SHA256,
    ETTH1_URL,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_etth1(path: Path, expected: str = ETTH1_SHA256) -> None:
    """Raise unless ``path`` is byte-for-byte the file the results came from."""

    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"{path} does not match the pinned dataset.\n"
            f"  expected sha256 {expected}\n"
            f"  found    sha256 {actual}\n"
            "The upstream URL tracks a branch, so the file may have been "
            "changed at source. Re-pin ETTH1_SHA256 in src/config.py only "
            "after deciding the new file is the one you want, and regenerate "
            "the results and the drift reference with it."
        )


def download_etth1(
    force: bool = False,
    dest_dir: Optional[Path] = None,
    verify_checksum: bool = True,
) -> Path:
    """Download ETTh1.csv into ``data/raw/`` if not already present."""

    dest_dir = dest_dir or RAW_DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ETTH1_FILENAME

    if dest.exists() and not force:
        if verify_checksum:
            verify_etth1(dest)
        return dest

    # certifi bundles up-to-date root certificates; using it avoids the
    # SSL verification failures Windows-bundled Python sometimes produces.
    response = requests.get(ETTH1_URL, timeout=60, verify=certifi.where())
    response.raise_for_status()
    dest.write_bytes(response.content)
    if verify_checksum:
        verify_etth1(dest)
    return dest


def load_etth1(
    target: str = TARGET_COLUMN,
    csv_path: Optional[Path] = None,
    download_if_missing: bool = True,
    verify_checksum: bool = True,
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
        csv_path = download_etth1(verify_checksum=verify_checksum)
    elif verify_checksum:
        verify_etth1(csv_path)

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


def make_synthetic(n_hours: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic hourly series for offline tests/CI.

    Combines a daily and weekly seasonality, a linear trend and Gaussian noise.
    Same column layout as ETTh1 so downstream code is exchangeable.
    """

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    daily = 5.0 * np.sin(2 * np.pi * t / 24)
    weekly = 2.0 * np.sin(2 * np.pi * t / (24 * 7))
    trend = 0.001 * t
    noise = rng.normal(0, 0.5, size=n_hours)
    ot = 20.0 + daily + weekly + trend + noise

    df = pd.DataFrame(
        {
            "HUFL": ot + rng.normal(0, 0.3, size=n_hours),
            "HULL": ot * 0.5 + rng.normal(0, 0.3, size=n_hours),
            "MUFL": ot + rng.normal(0, 0.2, size=n_hours),
            "MULL": ot * 0.4 + rng.normal(0, 0.2, size=n_hours),
            "LUFL": ot * 0.8 + rng.normal(0, 0.1, size=n_hours),
            "LULL": ot * 0.3 + rng.normal(0, 0.1, size=n_hours),
            "OT": ot,
        },
        index=idx,
    )
    df.index.name = DATE_COLUMN
    return df
