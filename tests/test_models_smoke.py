"""Lightweight smoke tests so CI catches breakage without internet/model downloads.

The Chronos model is intentionally NOT tested here — it requires the foundation
model weights, which we don't want to download in CI. The benchmark script
exercises it end-to-end locally.
"""

from __future__ import annotations

import pytest

from src.config import LightGBMConfig, LSTMConfig
from src.data import make_synthetic, temporal_split
from src.evaluation import compute_metrics
from src.models import LightGBMForecaster, LSTMForecaster


@pytest.fixture(scope="module")
def synthetic_splits():
    df = make_synthetic(n_hours=2000, seed=7)
    return temporal_split(df)


def test_lightgbm_runs_end_to_end(synthetic_splits) -> None:
    cfg = LightGBMConfig(n_estimators=80, num_leaves=15, early_stopping_rounds=20)
    model = LightGBMForecaster(config=cfg)
    model.fit(synthetic_splits.train, synthetic_splits.val)
    y_true, y_pred = model.predict(synthetic_splits.test)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.n > 0
    # Synthetic series is highly predictable -> RMSE should be small.
    assert metrics.rmse < 2.0


def test_lstm_runs_end_to_end(synthetic_splits) -> None:
    cfg = LSTMConfig(
        input_window=48,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        batch_size=32,
        epochs=2,
        early_stopping_patience=2,
    )
    model = LSTMForecaster(config=cfg)
    model.fit(synthetic_splits.train, synthetic_splits.val)
    y_true, y_pred = model.predict(synthetic_splits.test)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.n > 0
    # With only 2 epochs we don't assert quality, only that nothing crashed
    # and the prediction is finite.
    assert metrics.rmse > 0
