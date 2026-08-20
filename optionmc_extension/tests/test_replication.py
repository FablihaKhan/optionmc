"""Replication helpers and convergence-order fitting."""
import numpy as np
import pytest

from src.replication import (fit_convergence_order, independent_seeds,
                             replicate_lsmc, summarise)


def test_seeds_are_distinct_and_reproducible():
    a = independent_seeds(42, 30)
    b = independent_seeds(42, 30)
    assert a == b
    assert len(set(a)) == 30
    assert independent_seeds(43, 30) != a


def test_replications_differ_from_each_other_but_repeat_across_calls():
    kwargs = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
                  n_paths=2_000, n_steps=25, degree=2)
    first = replicate_lsmc(5, 42, **kwargs)
    second = replicate_lsmc(5, 42, **kwargs)

    assert np.array_equal(first["prices"], second["prices"])
    assert len(set(first["prices"])) == 5          # genuinely independent runs
    assert first["prices"].shape == (5,)
    assert np.all(first["runtimes"] > 0)


def test_replicate_rejects_zero_replications():
    with pytest.raises(ValueError):
        replicate_lsmc(0, 42, S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)


def test_summarise_separates_bias_from_spread():
    prices = np.array([10.0, 12.0, 11.0, 13.0])   # mean 11.5
    summary = summarise(prices, benchmark=10.0)

    assert summary["mean_price"] == pytest.approx(11.5)
    assert summary["bias"] == pytest.approx(1.5)
    assert summary["std_price"] == pytest.approx(np.std(prices, ddof=1))
    assert summary["mean_absolute_error"] == pytest.approx(1.5)
    assert summary["rmse"] == pytest.approx(np.sqrt(np.mean((prices - 10) ** 2)))
    assert summary["rmse"] >= abs(summary["bias"])     # RMSE contains the bias
    assert summary["std_error_of_mean"] == pytest.approx(
        summary["std_price"] / 2.0)
    assert summary["n_replications"] == 4


def test_summarise_handles_a_single_replication():
    summary = summarise(np.array([10.0]), benchmark=9.0)
    assert summary["bias"] == pytest.approx(1.0)
    assert np.isnan(summary["std_price"])
    assert np.isnan(summary["std_error_of_mean"])


def test_convergence_order_recovers_a_known_exponent():
    """Errors built as C * N^-0.5 must fit an order of exactly -0.5."""
    sizes = np.array([1_000, 5_000, 10_000, 25_000, 50_000], dtype=float)
    errors = 3.7 * sizes ** -0.5
    fit = fit_convergence_order(sizes, errors)

    assert fit["order"] == pytest.approx(-0.5, abs=1e-10)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-10)
    assert np.exp(fit["intercept"]) == pytest.approx(3.7, rel=1e-8)


def test_convergence_order_recovers_a_first_order_rate():
    sizes = np.array([10, 25, 50, 100], dtype=float)
    fit = fit_convergence_order(sizes, 2.0 / sizes)
    assert fit["order"] == pytest.approx(-1.0, abs=1e-10)


def test_convergence_order_rejects_impossible_input():
    with pytest.raises(ValueError):
        fit_convergence_order([100.0], [1.0])
    with pytest.raises(ValueError):
        fit_convergence_order([100.0, 200.0], [1.0])
    with pytest.raises(ValueError):
        fit_convergence_order([100.0, 200.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        fit_convergence_order([0.0, 200.0], [1.0, 0.5])


@pytest.mark.slow
def test_replicated_lsmc_spread_matches_the_theoretical_rate():
    """The run-to-run spread must fall like 1/sqrt(N), as theory predicts."""
    contract = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2, n_steps=25,
                    degree=2, antithetic=True)
    sizes = [2_000, 8_000, 32_000]
    spreads = [replicate_lsmc(20, 42, n_paths=n, **contract)["prices"].std(ddof=1)
               for n in sizes]
    fit = fit_convergence_order(sizes, spreads)
    assert fit["order"] == pytest.approx(-0.5, abs=0.12)
