"""Temporal feature engineering for tabular forecasters (LightGBM).

Produces three families of features:
- Lags of the target.
- Rolling statistics over multiple windows.
- Calendar features (hour, day-of-week, month, weekend flag, cyclical encodings).

All features are computed using ONLY past observations so there is no target
leakage when training a model to predict ``y[t+horizon]``.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, TARGET_COLUMN, FeatureConfig


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: Iterable[int],
    stats: Iterable[str],
) -> pd.DataFrame:
    out = df.copy()
    # shift(1) so the rolling window NEVER sees the current row.
    shifted = out[target].shift(1)
    for window in windows:
        roll = shifted.rolling(window=window, min_periods=max(2, window // 4))
        for stat in stats:
            if not hasattr(roll, stat):
                raise ValueError(f"Unsupported rolling stat: {stat}")
            out[f"{target}_roll_{stat}_{window}"] = getattr(roll, stat)()
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Calendar features require a DatetimeIndex.")
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["day"] = idx.day
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)

    # Cyclical encoding so the tree (or any model) sees that 23h is near 0h.
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    horizon: int = 1,
    config: FeatureConfig = CONFIG.features,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return ``(X, y, feature_names)`` ready for a tabular learner.

    ``y[t]`` corresponds to ``target[t + horizon]`` so the model predicts a
    future value strictly from past information.
    """

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    work = df[[target]].copy()
    work = add_lag_features(work, target=target, lags=config.lag_hours)
    work = add_rolling_features(
        work, target=target, windows=config.rolling_windows, stats=config.rolling_stats
    )
    if config.add_calendar:
        work = add_calendar_features(work)

    work[f"{target}_target"] = work[target].shift(-horizon)

    feature_cols = [
        c for c in work.columns if c not in (target, f"{target}_target")
    ]
    X = work[feature_cols]
    y = work[f"{target}_target"]

    if dropna:
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

    return X, y, feature_cols
