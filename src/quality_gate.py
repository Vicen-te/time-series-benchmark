"""Compare a fresh benchmark summary against the committed baseline.

A pipeline that retrains on every change but never checks the result will
happily publish a worse model. This module answers one question -- is the new
run allowed to replace the old one -- and is deliberately blunt about it: a
model that got worse by more than the tolerance, disappeared, or stopped being
scored on the same rows as its peers fails the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


LOWER_IS_BETTER = {"mae", "rmse", "mape"}


@dataclass(frozen=True)
class ModelDelta:
    model: str
    baseline: float
    candidate: float
    n: int

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def pct(self) -> float:
        if self.baseline == 0:
            return float("nan")
        return 100.0 * self.delta / abs(self.baseline)


@dataclass
class GateReport:
    metric: str
    tolerance: float
    deltas: List[ModelDelta] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_markdown(self) -> str:
        header = (
            f"| model | {self.metric} baseline | {self.metric} candidate | delta | |\n"
            "|:---|---:|---:|---:|:--:|\n"
        )
        rows = []
        for d in sorted(self.deltas, key=lambda x: x.candidate):
            ok = "FAIL" if self.is_regression(d) else "ok"
            rows.append(
                f"| {d.model} | {d.baseline:.4f} | {d.candidate:.4f} | "
                f"{d.pct:+.2f}% | {ok} |"
            )
        body = "\n".join(rows)
        verdict = "passed" if self.passed else "FAILED"
        tail = f"\n\nGate {verdict} (tolerance {self.tolerance:.1%} on {self.metric})."
        if self.failures:
            tail += "\n\n" + "\n".join(f"- {f}" for f in self.failures)
        return header + body + tail

    def baseline_limit(self, delta: ModelDelta) -> float:
        """Worst value ``delta.model`` may take and still pass."""

        if self.metric in LOWER_IS_BETTER:
            return delta.baseline * (1.0 + self.tolerance)
        return delta.baseline * (1.0 - self.tolerance)

    def is_regression(self, delta: ModelDelta) -> bool:
        limit = self.baseline_limit(delta)
        if self.metric in LOWER_IS_BETTER:
            return delta.candidate > limit
        return delta.candidate < limit


def _require_columns(df: pd.DataFrame, metric: str, label: str) -> None:
    for column in ("model", metric):
        if column not in df.columns:
            raise ValueError(f"{label} summary is missing the '{column}' column.")


def compare_summaries(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str = "rmse",
    tolerance: float = 0.03,
    require_equal_n: bool = True,
) -> GateReport:
    """Check ``candidate`` against ``baseline`` on ``metric``.

    ``tolerance`` is a fraction of the baseline value, so 0.03 lets an error
    metric grow by 3% before the gate trips -- enough headroom for the float
    noise of a different machine, far less than any regression worth shipping.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be >= 0.")
    _require_columns(baseline, metric, "baseline")
    _require_columns(candidate, metric, "candidate")

    report = GateReport(metric=metric, tolerance=tolerance)
    cand_by_model = candidate.set_index("model")
    base_by_model = baseline.set_index("model")

    for model, base_row in base_by_model.iterrows():
        if model not in cand_by_model.index:
            report.failures.append(f"{model}: present in the baseline, missing from this run")
            continue

        cand_row = cand_by_model.loc[model]
        delta = ModelDelta(
            model=str(model),
            baseline=float(base_row[metric]),
            candidate=float(cand_row[metric]),
            n=int(cand_row["n"]) if "n" in cand_row else -1,
        )
        report.deltas.append(delta)

        if report.is_regression(delta):
            report.failures.append(
                f"{model}: {metric} {delta.baseline:.4f} -> {delta.candidate:.4f} "
                f"({delta.pct:+.2f}%), past the {tolerance:.1%} tolerance"
            )

    if require_equal_n and "n" in candidate.columns:
        counts = sorted(set(int(v) for v in candidate["n"]))
        if len(counts) > 1:
            report.failures.append(
                "models were scored on different row counts "
                f"({counts}); errors measured over different windows are not comparable"
            )

    return report


def load_summary(path, metric: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if metric is not None:
        _require_columns(df, metric, str(path))
    return df
