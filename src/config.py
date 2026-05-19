"""Centralised configuration for the time-series benchmark.

All paths, splits, feature parameters and model hyperparameters live here so
that scripts stay thin and reproducible. Keep this file the single source of
truth.
"""

from __future__ import annotations

from pathlib import Path


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
