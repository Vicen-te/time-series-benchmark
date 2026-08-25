"""Is the gap between two forecasters real, or is it noise?

A benchmark table ranks models to four decimal places whether or not the
differences mean anything. On this dataset the spread between first and last
is under 6%, which is small enough that the ordering deserves a test rather
than a reader's trust.

Diebold-Mariano compares two sets of forecast errors on the same targets. It
tests the loss differential ``d[t] = loss(e_a[t]) - loss(e_b[t])`` against zero
with a Newey-West variance, so serial correlation in the errors -- which hourly
forecasts always have -- does not masquerade as significance. Implemented on
numpy so it carries no dependency the benchmark does not already have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


LOSSES = ("squared", "absolute")


def _normal_two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def newey_west_variance(series: np.ndarray, lags: int) -> float:
    """Long-run variance of the mean of ``series``, Bartlett-weighted."""

    n = series.size
    if n < 2:
        return float("nan")
    if lags < 0:
        raise ValueError("lags must be >= 0.")

    centred = series - series.mean()
    variance = float(centred @ centred) / n
    for k in range(1, min(lags, n - 1) + 1):
        gamma = float(centred[k:] @ centred[:-k]) / n
        variance += 2.0 * (1.0 - k / (lags + 1.0)) * gamma

    # Bartlett weights keep this non-negative in theory; guard the edge case
    # where rounding pushes it just below zero.
    return max(variance, 0.0) / n


def default_lags(n: int) -> int:
    """Standard Newey-West bandwidth rule, floor(4 * (n/100)^(2/9))."""

    if n <= 0:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int
    lags: int

    @property
    def significant(self) -> bool:
        return np.isfinite(self.p_value) and self.p_value < 0.05


def diebold_mariano(
    errors_a: Iterable[float],
    errors_b: Iterable[float],
    loss: str = "squared",
    lags: Optional[int] = None,
) -> DMResult:
    """Test whether ``a`` forecasts better than ``b``.

    A negative statistic means ``a`` has the lower loss. The p-value is
    two-sided, so it answers "are these two distinguishable at all".
    """

    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}.")

    a = np.asarray(list(errors_a), dtype=np.float64).ravel()
    b = np.asarray(list(errors_b), dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    if a.size < 2:
        raise ValueError("Need at least two paired errors.")

    if loss == "squared":
        differential = a**2 - b**2
    else:
        differential = np.abs(a) - np.abs(b)

    n = differential.size
    used_lags = default_lags(n) if lags is None else lags
    variance = newey_west_variance(differential, used_lags)

    mean_diff = float(differential.mean())
    if not np.isfinite(variance) or variance <= 0.0:
        # Identical forecasts leave nothing to distinguish.
        statistic = 0.0 if math.isclose(mean_diff, 0.0, abs_tol=1e-15) else math.inf
        p_value = 1.0 if statistic == 0.0 else 0.0
        return DMResult(statistic, p_value, mean_diff, n, used_lags)

    statistic = mean_diff / math.sqrt(variance)
    return DMResult(
        statistic=statistic,
        p_value=_normal_two_sided_p(statistic),
        mean_loss_diff=mean_diff,
        n=n,
        lags=used_lags,
    )


def pairwise_significance(
    y_true: pd.Series,
    predictions: Mapping[str, pd.Series],
    loss: str = "squared",
    lags: Optional[int] = None,
) -> pd.DataFrame:
    """Every pair of models, tested on the rows they share with ``y_true``."""

    frame_rows = []
    for name_a, name_b in combinations(list(predictions.keys()), 2):
        common = y_true.index
        for name in (name_a, name_b):
            common = common.intersection(predictions[name].index)
        truth = y_true.loc[common].to_numpy(dtype=np.float64)
        err_a = truth - predictions[name_a].loc[common].to_numpy(dtype=np.float64)
        err_b = truth - predictions[name_b].loc[common].to_numpy(dtype=np.float64)

        result = diebold_mariano(err_a, err_b, loss=loss, lags=lags)
        rmse_a = float(np.sqrt(np.mean(err_a**2)))
        rmse_b = float(np.sqrt(np.mean(err_b**2)))
        frame_rows.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "rmse_a": rmse_a,
                "rmse_b": rmse_b,
                "delta_rmse": rmse_a - rmse_b,
                "dm_statistic": result.statistic,
                "p_value": result.p_value,
                "n": result.n,
                "lags": result.lags,
                "verdict": "distinguishable" if result.significant else "not distinguishable",
            }
        )

    return pd.DataFrame(frame_rows)


def p_values_against(
    reference: str,
    y_true: pd.Series,
    predictions: Mapping[str, pd.Series],
    loss: str = "squared",
    lags: Optional[int] = None,
) -> Dict[str, float]:
    """p-value of every model against ``reference``, for the summary table."""

    if reference not in predictions:
        return {}

    pairs = pairwise_significance(y_true, predictions, loss=loss, lags=lags)
    out: Dict[str, float] = {reference: float("nan")}
    for row in pairs.itertuples():
        if row.model_a == reference:
            out[row.model_b] = row.p_value
        elif row.model_b == reference:
            out[row.model_a] = row.p_value
    return out


def to_markdown(pairs: pd.DataFrame) -> str:
    if pairs.empty:
        return "_Only one model in this run; nothing to compare._"

    header = "| pair | delta RMSE | DM | p | verdict |\n|:---|---:|---:|---:|:---|\n"
    body = "\n".join(
        f"| {row.model_a} vs {row.model_b} | {row.delta_rmse:+.4f} | "
        f"{row.dm_statistic:+.2f} | {row.p_value:.4f} | {row.verdict} |"
        for row in pairs.sort_values("p_value").itertuples()
    )
    return header + body
