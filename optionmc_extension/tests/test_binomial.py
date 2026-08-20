"""CRR binomial tree checks against published American-option values."""
import pytest

from src.binomial import (crr_american_put, crr_bermudan_put,
                          crr_european_put, early_exercise_premium)
from src.black_scholes import bs_put


# Longstaff & Schwartz (2001), Table 1, "Finite difference American" column.
# K = 40, r = 0.06, no dividends, continuously exercisable.
LS_TABLE1_AMERICAN_PUT = [
    # (S, sigma, T, published finite-difference American)
    (36.0, 0.20, 1, 4.478),
    (36.0, 0.20, 2, 4.840),
    (36.0, 0.40, 1, 7.101),
    (36.0, 0.40, 2, 8.508),
    (38.0, 0.20, 1, 3.250),
    (38.0, 0.40, 2, 7.670),
    (40.0, 0.20, 1, 2.314),
    (40.0, 0.40, 2, 6.920),
    (42.0, 0.20, 1, 1.617),
    (44.0, 0.20, 1, 1.110),
    (44.0, 0.40, 2, 5.647),
]


@pytest.mark.parametrize("S,sigma,T,expected", LS_TABLE1_AMERICAN_PUT)
def test_matches_longstaff_schwartz_finite_difference(S, sigma, T, expected):
    """A fine CRR tree must reproduce the paper's finite-difference values."""
    price = crr_american_put(S, K=40.0, T=T, r=0.06, sigma=sigma,
                             n_steps=5000)
    assert price == pytest.approx(expected, abs=0.01)


def test_european_tree_converges_to_black_scholes():
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.25, 0.015
    analytic = bs_put(S0, K, T, r, sigma, q)
    coarse = crr_european_put(S0, K, T, r, sigma, q, n_steps=50)
    fine = crr_european_put(S0, K, T, r, sigma, q, n_steps=5000)

    assert fine == pytest.approx(analytic, abs=0.005)
    assert abs(fine - analytic) < abs(coarse - analytic)


def test_american_dominates_european_and_intrinsic():
    """Scope section 10 sanity checks, on the benchmark itself."""
    S0, K, T, r, sigma = 35.0, 40.0, 1.0, 0.06, 0.2
    american = crr_american_put(S0, K, T, r, sigma, n_steps=2000)
    european = crr_european_put(S0, K, T, r, sigma, n_steps=2000)

    assert american >= european
    assert american >= max(K - S0, 0.0) - 1e-9
    assert american <= K


def test_early_exercise_premium_is_positive_for_american_put():
    premium = early_exercise_premium(36.0, 40.0, 1.0, 0.06, 0.2, n_steps=2000)
    assert premium > 0.5


def test_price_is_stable_as_steps_increase():
    """Scope section 10: the benchmark must settle as steps grow."""
    args = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
    prices = [crr_american_put(n_steps=n, **args)
              for n in (500, 1000, 2000, 4000)]
    assert abs(prices[-1] - prices[-2]) < 0.005


def test_price_falls_as_spot_rises():
    args = dict(K=40.0, T=1.0, r=0.06, sigma=0.2, n_steps=1000)
    prices = [crr_american_put(S0=s, **args) for s in (30.0, 36.0, 40.0, 46.0)]
    assert all(a > b for a, b in zip(prices, prices[1:]))


def test_rejects_degenerate_tree():
    with pytest.raises(ValueError):
        # Huge rate with tiny volatility pushes p outside (0, 1).
        crr_american_put(100.0, 100.0, 1.0, r=2.0, sigma=0.01, n_steps=5)


def test_american_implied_volatility_round_trips():
    """Price an American put at a known sigma, invert, recover it."""
    from src.binomial import implied_volatility_american_put

    for sigma in (0.12, 0.25, 0.45):
        price = crr_american_put(768.37, 749.0, 0.2, 0.0379, sigma, q=0.0121,
                                 n_steps=500)
        recovered = implied_volatility_american_put(
            price, 768.37, 749.0, 0.2, 0.0379, q=0.0121, n_steps=500)
        assert recovered == pytest.approx(sigma, abs=1e-6)


def test_american_implied_volatility_exceeds_european_inversion():
    """Inverting the European formula on an American quote understates sigma."""
    from src.binomial import implied_volatility_american_put
    from src.black_scholes import implied_volatility_put

    true_sigma = 0.30
    american_price = crr_american_put(36.0, 40.0, 1.0, 0.06, true_sigma,
                                      n_steps=1000)
    american_iv = implied_volatility_american_put(american_price, 36.0, 40.0,
                                                  1.0, 0.06, n_steps=1000)
    european_iv = implied_volatility_put(american_price, 36.0, 40.0, 1.0, 0.06)

    assert american_iv == pytest.approx(true_sigma, abs=1e-5)
    assert european_iv > american_iv


def test_bermudan_with_one_date_is_european_when_out_of_the_money():
    """One exercise date at maturity reduces to the European price.

    Time zero is itself an exercise opportunity -- as it is in the LSMC, whose
    price is max(continuation, intrinsic) -- so this identity only holds while
    immediate exercise is not already the better choice. The at-the-money case
    below has zero intrinsic value, so it isolates the maturity payoff.
    """
    args = dict(S0=40.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
    bermudan = crr_bermudan_put(n_steps=2000, n_exercise_dates=1, **args)
    european = crr_european_put(n_steps=2000, **args)
    assert bermudan == pytest.approx(european, abs=1e-10)


def test_bermudan_honours_immediate_exercise_at_time_zero():
    """Deep in the money, waiting is worse than exercising right now."""
    args = dict(S0=20.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
    bermudan = crr_bermudan_put(n_steps=2000, n_exercise_dates=1, **args)
    european = crr_european_put(n_steps=2000, **args)
    assert bermudan == pytest.approx(max(european, 40.0 - 20.0), abs=1e-10)
    assert bermudan == pytest.approx(20.0, abs=1e-10)


def test_bermudan_with_every_node_is_american():
    """Exercising at every tree node is exactly the American price."""
    args = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
    bermudan = crr_bermudan_put(n_steps=1000, n_exercise_dates=1000, **args)
    american = crr_american_put(n_steps=1000, **args)
    assert bermudan == pytest.approx(american, abs=1e-10)


def test_bermudan_price_rises_with_more_exercise_dates():
    """More exercise opportunities can only be worth more."""
    args = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.2)
    prices = [crr_bermudan_put(n_steps=2000, n_exercise_dates=m, **args)
              for m in (1, 10, 25, 50, 100, 200)]
    assert all(a <= b + 1e-12 for a, b in zip(prices, prices[1:]))
    american = crr_american_put(n_steps=2000, **args)
    assert prices[-1] <= american + 1e-12


def test_bermudan_rejects_incompatible_grids():
    with pytest.raises(ValueError, match="must be a multiple"):
        crr_bermudan_put(36.0, 40.0, 1.0, 0.06, 0.2, n_steps=100,
                         n_exercise_dates=30)
    with pytest.raises(ValueError):
        crr_bermudan_put(36.0, 40.0, 1.0, 0.06, 0.2, n_exercise_dates=0)
