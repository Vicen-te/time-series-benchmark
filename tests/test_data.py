"""Tests for the data loading and temporal split logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import TimeSeriesSplit, make_synthetic, temporal_split


def test_synthetic_dataset_has_expected_shape() -> None:
    df = make_synthetic(n_hours=200)
    assert len(df) == 200
    assert "OT" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_temporal_split_is_chronological_and_sums_to_total() -> None:
    df = make_synthetic(n_hours=1000)
    splits = temporal_split(df, train_frac=0.7, val_frac=0.15)
    sizes = splits.sizes
    assert sum(sizes.values()) == 1000
    assert sizes["train"] == 700
    assert sizes["val"] == 150
    assert sizes["test"] == 150

    assert splits.train.index.max() <= splits.val.index.min()
    assert splits.val.index.max() <= splits.test.index.min()


def test_temporal_split_rejects_invalid_fractions() -> None:
    df = make_synthetic(n_hours=100)
    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.6, val_frac=0.5)
    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.0, val_frac=0.5)


def test_split_describe_is_human_readable() -> None:
    df = make_synthetic(n_hours=300)
    splits = temporal_split(df)
    text = splits.describe()
    assert "train" in text and "val" in text and "test" in text
    assert isinstance(splits, TimeSeriesSplit)
