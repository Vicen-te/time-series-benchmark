"""Tests for the Diebold-Mariano comparison.

The point of the test is to stop a benchmark from claiming an ordering it
cannot support, so both mistakes matter: calling noise a win, and calling a
real gap noise. Both are pinned below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.significance import (
    default_lags,
    diebold_mariano,
    newey_west_variance,
    p_values_against,
    pairwise_significance,
    to_markdown,
)


@pytest.fixture()
def truth() -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.cumsum(rng.normal(size=1500))


# --- the statistic -------------------------------------------------------

def test_identical_forecasts_are_not_distinguishable(truth) -> None:
    rng = np.random.default_rng(1)
    errors = rng.normal(size=truth.size)
    result = diebold_mariano(errors, errors)
    assert result.mean_loss_diff == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant


def test_a_clearly_better_forecast_is_detected(truth) -> None:
    rng = np.random.default_rng(2)
    good = rng.normal(scale=0.5, size=truth.size)
    bad = rng.normal(scale=3.0, size=truth.size)
    result = diebold_mariano(good, bad)
    assert result.significant
    assert result.statistic < 0  # negative means the first argument wins
    assert result.p_value < 1e-6


def test_false_positive_rate_stays_near_the_nominal_five_percent() -> None:
    """Two equally good forecasters must be called apart about 5% of the time.

    Asserting on a single draw would only pin whichever seed happened to fall
    on the right side of 0.05 -- roughly one seed in twenty does not, which is
    the test working rather than failing. The property worth holding is the
    rejection rate over many draws.
    """

    rng = np.random.default_rng(3)
    trials = 300
    rejected = sum(
        diebold_mariano(
            rng.normal(scale=1.0, size=600),
            rng.normal(scale=1.0, size=600),
        ).significant
        for _ in range(trials)
    )
    rate = rejected / trials
    # Nominal 0.05; binomial sd over 300 trials is about 0.013, so this band is
    # wide enough never to flake and tight enough to catch a broken variance.
    assert 0.01 < rate < 0.12, f"rejection rate {rate:.3f} is not near 0.05"


def test_statistic_is_antisymmetric(truth) -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(scale=0.7, size=truth.size)
    b = rng.normal(scale=1.3, size=truth.size)
    forward = diebold_mariano(a, b)
    backward = diebold_mariano(b, a)
    assert forward.statistic == pytest.approx(-backward.statistic)
    assert forward.p_value == pytest.approx(backward.p_value)


def test_absolute_loss_is_supported(truth) -> None:
    rng = np.random.default_rng(5)
    good = rng.normal(scale=0.5, size=truth.size)
    bad = rng.normal(scale=3.0, size=truth.size)
    assert diebold_mariano(good, bad, loss="absolute").significant


def test_serial_correlation_widens_the_interval() -> None:
    """Autocorrelated errors must not be read as extra evidence.

    The same loss differential tested with no lag correction looks more
    significant than it is; the Newey-West terms are what stop that.
    """

    rng = np.random.default_rng(6)
    shock = rng.normal(size=4000)
    correlated = shock.copy()
    for i in range(1, correlated.size):
        correlated[i] += 0.9 * correlated[i - 1]

    naive = diebold_mariano(correlated + 0.3, np.zeros_like(correlated), lags=0)
    corrected = diebold_mariano(correlated + 0.3, np.zeros_like(correlated), lags=40)
    assert abs(corrected.statistic) < abs(naive.statistic)


def test_newey_west_with_no_lags_is_the_plain_variance_of_the_mean() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(size=500)
    expected = sample.var() / sample.size
    assert newey_west_variance(sample, lags=0) == pytest.approx(expected)


def test_default_lag_rule_grows_with_the_sample() -> None:
    assert default_lags(100) == 4
    assert default_lags(2613) > default_lags(100)
    assert default_lags(0) == 0


def test_invalid_inputs_raise(truth) -> None:
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        diebold_mariano([1.0], [2.0])
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0, 2.0], loss="ghost")
    with pytest.raises(ValueError):
        newey_west_variance(np.zeros(10), lags=-1)


# --- over a set of models ------------------------------------------------

def _predictions() -> tuple[pd.Series, dict[str, pd.Series]]:
    rng = np.random.default_rng(8)
    index = pd.date_range("2020-01-01", periods=1200, freq="h")
    y = pd.Series(np.cumsum(rng.normal(size=index.size)), index=index)
    return y, {
        "good": y + rng.normal(scale=0.3, size=index.size),
        "bad": y + rng.normal(scale=2.5, size=index.size),
        "twin": y + rng.normal(scale=0.3, size=index.size),
    }


def test_pairwise_covers_every_combination() -> None:
    y, preds = _predictions()
    pairs = pairwise_significance(y, preds)
    assert len(pairs) == 3
    assert set(pairs.columns) >= {"model_a", "model_b", "delta_rmse", "p_value", "verdict"}
    assert (pairs["n"] == len(y)).all()


def test_pairwise_separates_the_real_gap_from_the_twin() -> None:
    y, preds = _predictions()
    pairs = pairwise_significance(y, preds).set_index(["model_a", "model_b"])
    assert pairs.loc[("good", "bad"), "verdict"] == "distinguishable"
    assert pairs.loc[("good", "twin"), "verdict"] == "not distinguishable"


def test_p_values_against_a_reference() -> None:
    y, preds = _predictions()
    values = p_values_against("good", y, preds)
    assert np.isnan(values["good"])
    assert values["bad"] < 0.05
    assert values["twin"] > 0.05
    assert p_values_against("absent", y, preds) == {}


def test_pairwise_uses_only_shared_rows() -> None:
    y, preds = _predictions()
    preds["bad"] = preds["bad"].iloc[100:]
    pairs = pairwise_significance(y, preds).set_index(["model_a", "model_b"])
    assert pairs.loc[("good", "bad"), "n"] == len(y) - 100
    assert pairs.loc[("good", "twin"), "n"] == len(y)


def test_markdown_puts_the_strongest_evidence_first() -> None:
    y, preds = _predictions()
    md = to_markdown(pairwise_significance(y, preds))
    assert md.index("bad") < md.index("twin")
    assert to_markdown(pd.DataFrame()).startswith("_Only one model")
