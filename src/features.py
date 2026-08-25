"""Temporal feature engineering for tabular forecasters (LightGBM).

Produces three families of features:
- Lags of the target.
- Rolling statistics over multiple windows.
- Calendar features (hour, day-of-week, month, weekend flag, cyclical encodings).

Every row is labelled with the timestamp it predicts: ``y[t]`` is ``target[t]``
and every lag/rolling feature on that row is derived from observations no later
than ``t - horizon``. Calendar features are the exception and legitimately so --
the clock is known ahead of time.
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
    offset: int = 0,
) -> pd.DataFrame:
    """Lag ``target`` by ``offset + lag`` for each requested lag."""

    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(offset + lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: Iterable[int],
    stats: Iterable[str],
    shift: int = 1,
) -> pd.DataFrame:
    """Rolling stats over a window ending ``shift`` rows before the current one."""

    out = df.copy()
    shifted = out[target].shift(shift)
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

    ``y[t]`` is ``target[t]``, and the newest observation any lag or rolling
    feature on row ``t`` can reach is ``t - horizon``. A model trained on this
    matrix therefore forecasts exactly ``horizon`` steps past the last value it
    is allowed to see, which is what makes its error comparable to a sequence
    model fed the same history.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    work = df[[target]].copy()
    work = add_lag_features(
        work, target=target, lags=config.lag_hours, offset=horizon - 1
    )
    work = add_rolling_features(
        work,
        target=target,
        windows=config.rolling_windows,
        stats=config.rolling_stats,
        shift=horizon,
    )
    if config.add_calendar:
        work = add_calendar_features(work)

    feature_cols = [c for c in work.columns if c != target]
    X = work[feature_cols]
    y = work[target]

    if dropna:
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

    return X, y, feature_cols
