"""Tests for the temporal feature-engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FeatureConfig
from src.data import make_synthetic
from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
)


def test_lag_features_do_not_leak_future() -> None:
    df = make_synthetic(n_hours=200)
    out = add_lag_features(df, target="OT", lags=(1, 2))
    # First two rows must be NaN because the lag hasn't accumulated yet.
    assert out["OT_lag_1"].isna().sum() == 1
    assert out["OT_lag_2"].isna().sum() == 2
    # Values match the actual shifted series.
    assert out["OT_lag_1"].iloc[5] == pytest.approx(df["OT"].iloc[4])


def test_rolling_window_does_not_include_current_row() -> None:
    df = make_synthetic(n_hours=300)
    out = add_rolling_features(df, target="OT", windows=(24,), stats=("mean",))
    # Row i must be the mean of rows [i-24, i-1] (NOT including i).
    sample = out["OT_roll_mean_24"].iloc[50]
    expected = df["OT"].iloc[26:50].mean()
    assert sample == pytest.approx(expected, rel=1e-6)


def test_calendar_features_have_expected_columns_and_ranges() -> None:
    df = make_synthetic(n_hours=200)
    out = add_calendar_features(df)
    for col in ("hour", "dayofweek", "month", "is_weekend",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert col in out.columns
    assert out["hour"].between(0, 23).all()
    assert out["dayofweek"].between(0, 6).all()
    assert out["is_weekend"].isin([0, 1]).all()
    assert np.isclose(out["hour_sin"].pow(2) + out["hour_cos"].pow(2), 1.0).all()


def test_build_feature_matrix_aligns_target_with_horizon() -> None:
    df = make_synthetic(n_hours=500)
    cfg = FeatureConfig(lag_hours=(1, 2, 24), rolling_windows=(24,), rolling_stats=("mean",))
    X, y, feature_cols = build_feature_matrix(df, target="OT", horizon=1, config=cfg)
    assert len(X) == len(y)
    assert "OT_lag_1" in feature_cols
    assert "OT_roll_mean_24" in feature_cols
    assert "hour" in feature_cols
    # Target at index t is the original value at index t + 1.
    first_ts = y.index[0]
    expected = df["OT"].loc[first_ts + pd.Timedelta(hours=1)]
    assert y.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_build_feature_matrix_rejects_missing_target() -> None:
    df = make_synthetic(n_hours=100)
    with pytest.raises(KeyError):
        build_feature_matrix(df, target="ghost", horizon=1)
