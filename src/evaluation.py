"""Regression metrics and result aggregation for the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float
    mape: float
    n: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that ignores rows where y_true is too small to be meaningful."""

    eps = 1e-3
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Metrics:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = _safe_mape(y_true, y_pred)

    return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape, n=int(y_true.size))


def metrics_to_row(model_name: str, metrics: Metrics) -> Dict[str, float]:
    row = {"model": model_name}
    row.update(metrics.to_dict())
    return row


def results_to_dataframe(rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if not df.empty:
        df = df.sort_values("rmse").reset_index(drop=True)
    return df
