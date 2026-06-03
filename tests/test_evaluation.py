"""Tests for the evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation import compute_metrics, metrics_to_row, results_to_dataframe


def test_perfect_prediction_yields_zero_error_and_unit_r2() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y, y)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.n == 5


def test_metrics_match_hand_calculation() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.mae == pytest.approx(2.0 / 3.0)
    assert metrics.rmse == pytest.approx(math.sqrt(2.0 / 3.0))


def test_compute_metrics_validates_shape() -> None:
    with pytest.raises(ValueError):
        compute_metrics([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_results_dataframe_is_sorted_by_rmse() -> None:
    m1 = compute_metrics([1, 2, 3], [1.5, 2.5, 3.5])
    m2 = compute_metrics([1, 2, 3], [1, 2, 3])
    rows = [metrics_to_row("worse", m1), metrics_to_row("better", m2)]
    df = results_to_dataframe(rows)
    assert df.iloc[0]["model"] == "better"
    assert df.iloc[1]["model"] == "worse"
