"""Lightweight smoke tests so CI catches breakage without internet/model downloads.

Chronos' weights are never downloaded here -- 183 MB on every CI run buys very
little -- but the logic wrapped around them is exercised with a stub pipeline,
so the context handling and index alignment are covered even though the
foundation model itself is only run locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LightGBMConfig, LSTMConfig
from src.data import make_synthetic, temporal_split
from src.evaluation import compute_metrics
from src.models import LightGBMForecaster, LSTMForecaster, PersistenceForecaster


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


def test_persistence_repeats_the_last_observation(synthetic_splits) -> None:
    model = PersistenceForecaster()
    model.fit(synthetic_splits.train, synthetic_splits.val)
    y_true, y_pred = model.predict(synthetic_splits.test)

    test = synthetic_splits.test
    # Every row is predicted with the value one step before it.
    assert len(y_true) == len(test) - 1
    assert y_true.index[0] == test.index[1]
    assert y_pred.iloc[0] == pytest.approx(test["OT"].iloc[0])
    assert y_true.iloc[0] == pytest.approx(test["OT"].iloc[1])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics.n == len(test) - 1
    assert metrics.rmse > 0


def test_persistence_needs_no_training(synthetic_splits) -> None:
    # The rule is fixed, so predicting straight after construction is valid.
    _, y_pred = PersistenceForecaster().predict(synthetic_splits.test)
    assert y_pred.notna().all()


def test_every_model_is_scored_on_the_same_rows(synthetic_splits) -> None:
    """Warm-up must come from the preceding split, not from the test slice.

    Taken from inside the slice it silently shortens each model's window by a
    different amount -- the deeper the lag or the longer the input window, the
    later that model starts -- and the summary table then compares errors
    measured over different periods.
    """

    train, val, test = synthetic_splits.train, synthetic_splits.val, synthetic_splits.test
    context = pd.concat([train, val])

    models = [
        PersistenceForecaster(),
        LightGBMForecaster(config=LightGBMConfig(n_estimators=40, early_stopping_rounds=10)),
        LSTMForecaster(
            config=LSTMConfig(input_window=48, hidden_size=8, num_layers=1,
                              dropout=0.0, epochs=1, early_stopping_patience=1)
        ),
    ]

    for model in models:
        model.fit(train, val)
        y_true, y_pred = model.predict(test, context=context)
        assert y_true.index.equals(test.index), f"{model.name} skipped rows"
        assert y_pred.index.equals(test.index), f"{model.name} skipped rows"


def test_missing_context_shortens_the_scored_window(synthetic_splits) -> None:
    train, val, test = synthetic_splits.train, synthetic_splits.val, synthetic_splits.test
    model = PersistenceForecaster()
    model.fit(train, val)

    without, _ = model.predict(test)
    with_context, _ = model.predict(test, context=pd.concat([train, val]))
    assert len(without) == len(test) - model.context_rows
    assert len(with_context) == len(test)


def test_context_must_precede_the_scored_frame(synthetic_splits) -> None:
    model = PersistenceForecaster()
    with pytest.raises(ValueError):
        model.predict(synthetic_splits.test, context=synthetic_splits.test)


class _LastValuePipeline:
    """Stub Chronos pipeline that forecasts the last value of each context.

    Chronos' own weights are a 183 MB download, but the code around them --
    stitching the warm-up context on, walking the rolling window, collapsing
    the quantile axis and labelling the result -- is ordinary logic that was
    getting no coverage at all. With a pipeline whose forecast is "repeat the
    last observation", the forecaster must reproduce the persistence baseline
    exactly; anything off by one row or one hour shows up immediately.
    """

    def predict(self, context, prediction_length):
        import torch

        last = torch.stack([torch.as_tensor(c)[-1] for c in context]).to(torch.float32)
        # (batch, quantiles, horizon), the shape chronos-bolt returns.
        return last.view(-1, 1, 1).repeat(1, 3, prediction_length)


def test_chronos_context_handling_reproduces_persistence(synthetic_splits) -> None:
    from src.models import ChronosForecaster

    train, val, test = synthetic_splits.train, synthetic_splits.val, synthetic_splits.test
    context = pd.concat([train, val])

    chronos = ChronosForecaster()
    chronos.pipeline_ = _LastValuePipeline()  # no weights, no download
    y_true, y_pred = chronos.predict(test, context=context)

    baseline = PersistenceForecaster()
    baseline.fit(train, val)
    base_true, base_pred = baseline.predict(test, context=context)

    assert y_true.index.equals(test.index)
    assert y_true.index.equals(base_true.index)
    assert np.allclose(y_true.to_numpy(), base_true.to_numpy(), atol=1e-4)
    assert np.allclose(y_pred.to_numpy(), base_pred.to_numpy(), atol=1e-4)


def test_chronos_without_context_cannot_score_the_earliest_rows(synthetic_splits) -> None:
    from src.models import ChronosForecaster

    chronos = ChronosForecaster()
    chronos.pipeline_ = _LastValuePipeline()
    y_true, _ = chronos.predict(synthetic_splits.test)

    # The first rows have too little history to form a window, so they are
    # dropped -- which is exactly why the benchmark passes a context in.
    assert len(y_true) < len(synthetic_splits.test)
    assert y_true.index[0] > synthetic_splits.test.index[0]


def test_chronos_requires_a_loaded_pipeline(synthetic_splits) -> None:
    from src.models import ChronosForecaster

    with pytest.raises(RuntimeError):
        ChronosForecaster().predict(synthetic_splits.test)

def test_build_model_propagates_the_horizon_to_every_model() -> None:
    """Every model the runner builds must forecast the horizon that was asked for.

    A model that quietly keeps its default of 1 while the others move to 24
    would still fill its row in the summary table, so the benchmark would
    compare a one-step forecaster against multi-step ones and report the
    mismatch as skill. Nothing else in the pipeline can catch that.
    """
    import importlib

    runner = importlib.import_module("scripts.run_benchmark")

    for name in runner.MODEL_FACTORIES:
        model = runner.build_model(name, device="cpu", horizon=24)
        assert model.horizon == 24, name
