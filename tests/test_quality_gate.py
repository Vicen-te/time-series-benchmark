"""Tests for the benchmark regression gate.

A gate that is wrong is worse than no gate: a false pass ships the regression
it exists to stop, a false failure gets switched off within a week. Both
directions are pinned here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.quality_gate import compare_summaries


def _summary(**overrides) -> pd.DataFrame:
    base = {
        "model": ["persistence", "lightgbm", "lstm"],
        "rmse": [0.6589, 0.6909, 0.6899],
        "mae": [0.4448, 0.4837, 0.4798],
        "r2": [0.9461, 0.9408, 0.9409],
        "n": [2613, 2613, 2613],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_identical_run_passes() -> None:
    report = compare_summaries(_summary(), _summary(), tolerance=0.0)
    assert report.passed
    assert report.failures == []
    assert len(report.deltas) == 3


def test_improvement_passes() -> None:
    better = _summary(rmse=[0.6589, 0.6000, 0.6100])
    report = compare_summaries(_summary(), better, tolerance=0.03)
    assert report.passed
    assert all(d.pct <= 0 for d in report.deltas)


def test_regression_beyond_tolerance_fails() -> None:
    worse = _summary(rmse=[0.6589, 0.9886, 0.6899])
    report = compare_summaries(_summary(), worse, tolerance=0.03)
    assert not report.passed
    assert len(report.failures) == 1
    assert "lightgbm" in report.failures[0]


def test_regression_inside_tolerance_passes() -> None:
    # +2% on a 3% tolerance: the float noise of a different machine.
    drifted = _summary(rmse=[0.6589, 0.6909 * 1.02, 0.6899])
    assert compare_summaries(_summary(), drifted, tolerance=0.03).passed
    assert not compare_summaries(_summary(), drifted, tolerance=0.01).passed


def test_missing_model_fails() -> None:
    partial = pd.DataFrame({"model": ["persistence"], "rmse": [0.6589], "n": [2613]})
    report = compare_summaries(_summary(), partial, tolerance=0.03)
    assert not report.passed
    assert any("lightgbm" in f and "missing" in f for f in report.failures)


def test_unequal_row_counts_fail() -> None:
    """The window bug this benchmark used to have must not come back quietly."""

    mismatched = _summary(n=[2613, 2444, 2517])
    report = compare_summaries(_summary(), mismatched, tolerance=0.03)
    assert not report.passed
    assert any("different row counts" in f for f in report.failures)
    assert compare_summaries(_summary(), mismatched, tolerance=0.03,
                             require_equal_n=False).passed


def test_higher_is_better_metric_flips_the_comparison() -> None:
    worse = _summary(r2=[0.9461, 0.5000, 0.9409])
    assert not compare_summaries(_summary(), worse, metric="r2", tolerance=0.03).passed

    better = _summary(r2=[0.9461, 0.9900, 0.9409])
    assert compare_summaries(_summary(), better, metric="r2", tolerance=0.03).passed


def test_markdown_report_marks_the_failing_row() -> None:
    worse = _summary(rmse=[0.6589, 0.9886, 0.6899])
    md = compare_summaries(_summary(), worse, tolerance=0.03).to_markdown()
    assert "FAIL" in md
    assert "Gate FAILED" in md
    assert md.count("| ok |") == 2


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        compare_summaries(_summary(), _summary(), tolerance=-0.1)
    with pytest.raises(ValueError):
        compare_summaries(_summary(), _summary(), metric="ghost")
