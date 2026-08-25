"""Tests for the data loading and temporal split logic."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from src.data import (
    TimeSeriesSplit,
    load_etth1,
    make_synthetic,
    sha256_of,
    temporal_split,
    verify_etth1,
)


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


def test_sha256_matches_hashlib(tmp_path) -> None:
    payload = b"some,csv\n1,2\n"
    target = tmp_path / "sample.csv"
    target.write_bytes(payload)
    assert sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_verify_accepts_the_expected_digest(tmp_path) -> None:
    target = tmp_path / "sample.csv"
    target.write_bytes(b"payload")
    verify_etth1(target, expected=sha256_of(target))


def test_verify_rejects_a_changed_file(tmp_path) -> None:
    """The upstream URL tracks a branch, so the file can change under us."""

    target = tmp_path / "sample.csv"
    target.write_bytes(b"payload")
    digest = sha256_of(target)
    target.write_bytes(b"payload tampered with")

    with pytest.raises(ValueError) as excinfo:
        verify_etth1(target, expected=digest)
    assert "does not match the pinned dataset" in str(excinfo.value)


def test_load_can_skip_verification(tmp_path) -> None:
    csv = tmp_path / "fake.csv"
    csv.write_text("date,OT\n2020-01-01 00:00:00,1.5\n2020-01-01 01:00:00,2.5\n")
    df = load_etth1(csv_path=csv, verify_checksum=False)
    assert list(df.columns) == ["OT"]
    assert len(df) == 2

    with pytest.raises(ValueError):
        load_etth1(csv_path=csv, verify_checksum=True)
