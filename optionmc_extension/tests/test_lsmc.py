"""Least-Squares Monte Carlo checks.

The first tests are the decisive ones: they rerun the eight-path worked example
from section 1 of Longstaff & Schwartz (2001) and must reproduce the paper's
published option value of 0.1144, its stopping rule, and its regression
coefficients.
"""
import numpy as np
import pytest

from src.binomial import crr_american_put
from src.european_mc import european_put_from_paths
from src.gbm import risk_neutral_drift, simulate_gbm_paths
from src.lsmc import lsmc_american_put_from_paths, price_american_put_lsmc


# Longstaff & Schwartz (2001), section 1: the eight stock-price paths used to
# illustrate the algorithm. Strike 1.10, riskless rate 6%, three annual dates.
PAPER_PATHS = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34],
])


def test_reproduces_paper_eight_path_example():
    """The paper reports 0.1144 American and 0.0564 European for this example."""
    result = lsmc_american_put_from_paths(PAPER_PATHS, K=1.10, r=0.06, dt=1.0,
                                          degree=2)
    assert result.price == pytest.approx(0.1144, abs=0.0001)
    assert result.european_price == pytest.approx(0.0564, abs=0.0001)
    assert result.early_exercise_premium > 0


def test_reproduces_paper_stopping_rule():
    """Paper: exercise at t=1 on paths 4, 6, 7, 8; path 3 waits until t=3."""
    result = lsmc_american_put_from_paths(PAPER_PATHS, K=1.10, r=0.06, dt=1.0,
                                          degree=2)
    # zero-based path indices 3, 5, 6, 7 stop at step 1
    assert list(result.stopping_step) == [3, 3, 3, 1, 3, 1, 1, 1]


def test_regression_matches_paper_coefficients():
    """Time-2 conditional expectation in the paper: -1.070 + 2.983X - 1.813X^2."""
    result = lsmc_american_put_from_paths(PAPER_PATHS, K=1.10, r=0.06, dt=1.0,
                                          degree=2)
    K = 1.10
    # Our betas are in normalised coordinates x = S/K with y = cash/K, so the
    # paper's coefficient on S^j is beta_j * K / K**j.
    beta = result.coefficients[2]
    a = [beta[j] * K / K ** j for j in range(3)]
    assert a[0] == pytest.approx(-1.070, abs=0.002)
    assert a[1] == pytest.approx(2.983, abs=0.002)
    assert a[2] == pytest.approx(-1.813, abs=0.002)


def test_only_in_the_money_paths_enter_the_regression():
    """Paper section 2.2: five paths are in the money at times 1 and 2."""
    result = lsmc_american_put_from_paths(PAPER_PATHS, K=1.10, r=0.06, dt=1.0,
                                          degree=2)
    assert result.n_itm[2] == 5
    assert result.n_itm[1] == 5


@pytest.mark.slow
@pytest.mark.parametrize("S,sigma,T", [
    (36.0, 0.20, 1),
    (40.0, 0.20, 1),
    (44.0, 0.20, 1),
    (36.0, 0.40, 1),
])
def test_lsmc_agrees_with_binomial_benchmark(S, sigma, T):
    """LSMC must land on the CRR benchmark, as in the paper's Table 1."""
    benchmark = crr_american_put(S, 40.0, T, 0.06, sigma, n_steps=5000)
    result = price_american_put_lsmc(
        S0=S, K=40.0, T=T, r=0.06, sigma=sigma,
        n_paths=100_000, n_steps=50 * T, degree=3, seed=42, antithetic=True)
    assert result.price == pytest.approx(benchmark, abs=0.05)
    assert result.std_error < 0.02


def test_american_is_never_worse_than_european_on_same_paths():
    """Scope section 10, with shared random draws so the comparison is exact."""
    rng = np.random.default_rng(5)
    paths = simulate_gbm_paths(38.0, risk_neutral_drift(0.06), 0.2, 1.0,
                               n_steps=50, n_paths=20_000, rng=rng,
                               antithetic=True)
    result = lsmc_american_put_from_paths(paths, K=40.0, r=0.06, dt=1.0 / 50)
    same_paths_european = european_put_from_paths(paths, K=40.0, r=0.06, T=1.0)

    assert result.price >= result.european_price
    assert result.european_price == pytest.approx(
        same_paths_european["price"], rel=1e-12)


def test_price_respects_intrinsic_and_strike_bounds():
    """Scope section 10: intrinsic <= price <= K."""
    for S0 in (25.0, 36.0, 40.0, 55.0):
        result = price_american_put_lsmc(
            S0=S0, K=40.0, T=1.0, r=0.06, sigma=0.2,
            n_paths=20_000, n_steps=50, seed=1)
        assert result.price >= max(40.0 - S0, 0.0) - 1e-9
        assert result.price <= 40.0


@pytest.mark.slow
def test_price_stabilises_as_paths_increase():
    """Scope section 10: raising N must settle the price."""
    benchmark = crr_american_put(36.0, 40.0, 1.0, 0.06, 0.2, n_steps=5000)
    errors = []
    for n_paths in (1_000, 10_000, 100_000):
        result = price_american_put_lsmc(
            S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
            n_paths=n_paths, n_steps=50, degree=3, seed=42, antithetic=True)
        errors.append(abs(result.price - benchmark))
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.05


def test_standard_error_shrinks_like_one_over_sqrt_n():
    a = price_american_put_lsmc(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
                                n_paths=10_000, n_steps=50, seed=9)
    b = price_american_put_lsmc(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
                                n_paths=40_000, n_steps=50, seed=9)
    ratio = a.std_error / b.std_error
    assert ratio == pytest.approx(2.0, rel=0.25)


def test_early_exercise_is_actually_happening():
    """An in-the-money American put must exercise early on many paths."""
    result = price_american_put_lsmc(S0=34.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
                                     n_paths=20_000, n_steps=50, seed=3)
    assert result.early_exercise_fraction > 0.3
    boundary = result.exercise_boundary
    assert np.isfinite(boundary).sum() > 10
    assert np.nanmax(boundary) <= 40.0      # exercise only when in the money


def test_reproducible_with_seed():
    kwargs = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2,
                  n_paths=5_000, n_steps=25, seed=77)
    first = price_american_put_lsmc(**kwargs).price
    second = price_american_put_lsmc(**kwargs).price
    assert first == second


def test_dividend_yield_is_applied_through_risk_neutral_drift():
    """Higher q lowers the forward, so an American put is worth more."""
    no_div = price_american_put_lsmc(S0=40.0, K=40.0, T=1.0, r=0.06,
                                     sigma=0.2, q=0.0, n_paths=50_000,
                                     n_steps=50, seed=21, antithetic=True)
    with_div = price_american_put_lsmc(S0=40.0, K=40.0, T=1.0, r=0.06,
                                       sigma=0.2, q=0.04, n_paths=50_000,
                                       n_steps=50, seed=21, antithetic=True)
    assert with_div.price > no_div.price


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        lsmc_american_put_from_paths(PAPER_PATHS[:, :1], K=1.1, r=0.06, dt=1.0)
    with pytest.raises(ValueError):
        lsmc_american_put_from_paths(PAPER_PATHS, K=1.1, r=0.06, dt=1.0,
                                     degree=0)
