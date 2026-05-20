"""Centralised configuration for the time-series benchmark.

All paths, splits, feature parameters and model hyperparameters live here so
that scripts stay thin and reproducible. Keep this file the single source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

for _p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, MLRUNS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ETTh1: Electricity Transformer Temperature dataset (hourly resolution).
# Standard benchmark used by Informer/Autoformer papers. ~17K hourly rows.
ETTH1_URL: str = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
)
ETTH1_FILENAME: str = "ETTh1.csv"

TARGET_COLUMN: str = "OT"  # Oil Temperature
DATE_COLUMN: str = "date"


@dataclass(frozen=True)
class SplitConfig:
    """Temporal 70/15/15 split. Time-series splits MUST be chronological."""

    train_frac: float = 0.70
    val_frac: float = 0.15

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering parameters for the tabular (LightGBM) pipeline."""

    lag_hours: Tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows: Tuple[int, ...] = (24, 48, 168)
    rolling_stats: Tuple[str, ...] = ("mean", "std", "min", "max")
    add_calendar: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    horizon: int = 1
    random_state: int = 42


CONFIG = BenchmarkConfig()
