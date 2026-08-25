"""Distribution drift between a reference sample and a candidate one.

Two complementary statistics, both computed without scipy so the check runs
wherever the benchmark runs:

- **PSI** (population stability index) bins the reference into quantiles and
  measures how much probability mass moved. It is the usual number in
  production monitoring, and its conventional bands are baked in below.
- **KS**, the largest gap between the two empirical CDFs. It notices shape
  changes that leave the binned mass roughly where it was.

Note what this can and cannot tell you here. ETTh1 is a fixed CSV, so the
train-to-test drift it reports is a real and permanent property of the series
-- the 2018 test window is genuinely not distributed like the 2016-2017
training window -- not a live signal that something broke today. Watching that
number for *movement* is still worth doing: it only changes if the data or the
feature pipeline changed, which is exactly the kind of silent breakage that is
otherwise found by reading metrics and guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


# Conventional PSI reading: below 0.10 the population is stable, up to 0.25 it
# has moved enough to look at, above that it has moved enough to act on.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def _clean(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    arr = arr.ravel()
    return arr[np.isfinite(arr)]


def population_stability_index(
    reference: Iterable[float],
    candidate: Iterable[float],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI of ``candidate`` against ``reference``, using reference quantile bins."""

    if bins < 2:
        raise ValueError("bins must be >= 2.")

    ref = _clean(reference)
    cand = _clean(candidate)
    if ref.size == 0 or cand.size == 0:
        return float("nan")

    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        # A constant reference has no distribution to move away from.
        return 0.0 if np.allclose(cand, ref[0]) else float("inf")

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cand_share = np.histogram(cand, bins=edges)[0] / cand.size

    ref_share = np.clip(ref_share, epsilon, None)
    cand_share = np.clip(cand_share, epsilon, None)

    return float(np.sum((cand_share - ref_share) * np.log(cand_share / ref_share)))


def ks_statistic(reference: Iterable[float], candidate: Iterable[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the CDFs."""

    ref = np.sort(_clean(reference))
    cand = np.sort(_clean(candidate))
    if ref.size == 0 or cand.size == 0:
        return float("nan")

    grid = np.concatenate([ref, cand])
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_cand = np.searchsorted(cand, grid, side="right") / cand.size
    return float(np.max(np.abs(cdf_ref - cdf_cand)))


def psi_band(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks: float

    @property
    def band(self) -> str:
        return psi_band(self.psi)


def drift_report(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Per-column PSI and KS, sorted by PSI descending."""

    if columns is None:
        columns = [
            c for c in reference.columns
            if c in candidate.columns and pd.api.types.is_numeric_dtype(reference[c])
        ]
    missing = [c for c in columns if c not in candidate.columns]
    if missing:
        raise KeyError(f"Columns missing from the candidate frame: {missing}")

    rows: List[FeatureDrift] = [
        FeatureDrift(
            feature=str(column),
            psi=population_stability_index(reference[column], candidate[column], bins=bins),
            ks=ks_statistic(reference[column], candidate[column]),
        )
        for column in columns
    ]

    frame = pd.DataFrame(
        [{"feature": r.feature, "psi": r.psi, "ks": r.ks, "band": r.band} for r in rows]
    )
    if frame.empty:
        return frame
    return frame.sort_values("psi", ascending=False).reset_index(drop=True)


def compare_to_reference(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance: float = 0.05,
) -> List[str]:
    """Flag features whose PSI moved from the recorded value by more than ``tolerance``.

    On a fixed dataset the PSI of a given feature is a constant. Movement means
    the data or the feature pipeline changed, so it is reported as a failure
    rather than as drift.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")

    failures: List[str] = []
    recorded = reference.set_index("feature")["psi"].to_dict()
    measured = current.set_index("feature")["psi"].to_dict()

    for feature, expected in recorded.items():
        if feature not in measured:
            failures.append(f"{feature}: recorded in the reference, missing from this run")
            continue
        moved = abs(measured[feature] - expected)
        if moved > tolerance:
            failures.append(
                f"{feature}: PSI {expected:.4f} -> {measured[feature]:.4f} "
                f"(moved {moved:.4f}, tolerance {tolerance:.4f})"
            )

    for feature in measured:
        if feature not in recorded:
            failures.append(f"{feature}: not in the reference, so its PSI is unverified")

    return failures


def to_markdown(report: pd.DataFrame, top: int = 15) -> str:
    if report.empty:
        return "_No comparable numeric columns._"

    header = "| feature | PSI | KS | band |\n|:---|---:|---:|:---|\n"
    body = "\n".join(
        f"| {row.feature} | {row.psi:.4f} | {row.ks:.4f} | {row.band} |"
        for row in report.head(top).itertuples()
    )
    tail = ""
    if len(report) > top:
        tail = f"\n\n_{len(report) - top} further columns omitted._"
    return header + body + tail
