"""Tests for the drift statistics.

ETTh1 never changes, so a drift check run against it can only ever return the
same answer. That makes a green tick meaningless on its own: the only way to
know the detector works is to hand it drift it has to find. These tests inject
known shifts and assert they are caught, and assert that an unshifted sample
stays quiet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import make_synthetic
from src.drift import (
    PSI_SIGNIFICANT,
    PSI_STABLE,
    compare_to_reference,
    drift_report,
    ks_statistic,
    population_stability_index,
    psi_band,
    to_markdown,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --- the statistics themselves -------------------------------------------

def test_identical_samples_show_no_drift(rng) -> None:
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)
    assert ks_statistic(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_different_draws_stays_below_the_stable_band(rng) -> None:
    a = rng.normal(size=8000)
    b = rng.normal(size=8000)
    assert population_stability_index(a, b) < PSI_STABLE
    assert psi_band(population_stability_index(a, b)) == "stable"


def test_disjoint_samples_saturate_ks(rng) -> None:
    a = rng.normal(size=1000)
    b = rng.normal(size=1000) + 1000.0
    assert ks_statistic(a, b) == pytest.approx(1.0)


@pytest.mark.parametrize("shift", [0.5, 1.0, 2.0, 4.0])
def test_psi_grows_with_the_size_of_the_injected_shift(rng, shift: float) -> None:
    reference = rng.normal(size=8000)
    small = population_stability_index(reference, reference + 0.25)
    injected = population_stability_index(reference, reference + shift)
    assert injected > small
    if shift >= 1.0:
        assert injected > PSI_SIGNIFICANT


def test_psi_catches_a_variance_only_shift(rng) -> None:
    """The mean is unchanged, so anything watching the mean would miss this."""

    reference = rng.normal(size=8000)
    widened = reference * 3.0
    assert np.isclose(widened.mean(), reference.mean(), atol=0.1)
    assert population_stability_index(reference, widened) > PSI_SIGNIFICANT
    assert ks_statistic(reference, widened) > 0.2


def test_constant_reference_is_handled(rng) -> None:
    flat = np.full(500, 7.0)
    assert population_stability_index(flat, flat) == 0.0
    assert population_stability_index(flat, flat + 1.0) == float("inf")


def test_empty_input_yields_nan() -> None:
    assert np.isnan(population_stability_index([], [1.0, 2.0]))
    assert np.isnan(ks_statistic([1.0, 2.0], []))


def test_invalid_bins_rejected(rng) -> None:
    with pytest.raises(ValueError):
        population_stability_index(rng.normal(size=10), rng.normal(size=10), bins=1)


# --- the report over a frame ---------------------------------------------

def test_report_flags_only_the_drifted_column(rng) -> None:
    reference = pd.DataFrame(
        {"quiet": rng.normal(size=6000), "drifted": rng.normal(size=6000)}
    )
    candidate = pd.DataFrame(
        {"quiet": rng.normal(size=6000), "drifted": rng.normal(size=6000) + 3.0}
    )

    report = drift_report(reference, candidate)
    bands = report.set_index("feature")["band"].to_dict()
    assert bands["drifted"] == "significant"
    assert bands["quiet"] == "stable"
    # Sorted worst-first so the offender is the row a reader sees.
    assert report.iloc[0]["feature"] == "drifted"


def test_report_on_a_shifted_synthetic_series() -> None:
    """End to end on the same generator the model smoke tests use."""

    df = make_synthetic(n_hours=4000, seed=3)
    first, second = df.iloc[:2000], df.iloc[2000:].copy()

    undrifted = drift_report(first, second, columns=["OT"])
    second["OT"] = second["OT"] + 12.0
    drifted = drift_report(first, second, columns=["OT"])

    assert drifted.iloc[0]["psi"] > undrifted.iloc[0]["psi"]
    assert drifted.iloc[0]["band"] == "significant"


def test_report_rejects_missing_columns(rng) -> None:
    reference = pd.DataFrame({"a": rng.normal(size=100)})
    candidate = pd.DataFrame({"b": rng.normal(size=100)})
    with pytest.raises(KeyError):
        drift_report(reference, candidate, columns=["a"])


def test_markdown_lists_the_worst_offender_first(rng) -> None:
    reference = pd.DataFrame({"quiet": rng.normal(size=3000), "drifted": rng.normal(size=3000)})
    candidate = pd.DataFrame({"quiet": rng.normal(size=3000), "drifted": rng.normal(size=3000) + 3.0})
    md = to_markdown(drift_report(reference, candidate))
    assert md.index("drifted") < md.index("quiet")
    assert to_markdown(pd.DataFrame()).startswith("_No comparable")


# --- comparison against the recorded reference ---------------------------

def _reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": ["a", "b"], "psi": [0.0100, 4.5000], "ks": [0.02, 0.61], "band": ["stable", "significant"]}
    )


def test_unmoved_psi_passes() -> None:
    assert compare_to_reference(_reference_frame(), _reference_frame(), tolerance=0.02) == []


def test_moved_psi_is_reported() -> None:
    moved = _reference_frame()
    moved.loc[moved.feature == "b", "psi"] = 6.0
    failures = compare_to_reference(moved, _reference_frame(), tolerance=0.02)
    assert len(failures) == 1 and "b:" in failures[0]


def test_missing_and_unexpected_features_are_reported() -> None:
    shrunk = _reference_frame().iloc[:1]
    assert any("missing" in f for f in compare_to_reference(shrunk, _reference_frame()))

    grown = pd.concat(
        [_reference_frame(), pd.DataFrame({"feature": ["c"], "psi": [0.01], "ks": [0.0], "band": ["stable"]})],
        ignore_index=True,
    )
    assert any("unverified" in f for f in compare_to_reference(grown, _reference_frame()))


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        compare_to_reference(_reference_frame(), _reference_frame(), tolerance=-1.0)
