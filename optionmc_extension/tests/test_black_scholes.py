"""Black-Scholes checks against published values."""
import numpy as np
import pytest

from src.black_scholes import bs_call, bs_put, bs_prices


# Longstaff & Schwartz (2001), Table 1, "Closed form European" column.
# K = 40, r = 0.06, no dividends.
LS_TABLE1_EUROPEAN_PUT = [
    # (S, sigma, T, published European put)
    (36.0, 0.20, 1, 3.844),
    (36.0, 0.20, 2, 3.763),
    (36.0, 0.40, 1, 6.711),
    (36.0, 0.40, 2, 7.700),
    (38.0, 0.20, 1, 2.852),
    (40.0, 0.20, 1, 2.066),
    (40.0, 0.40, 2, 6.326),
    (42.0, 0.20, 2, 1.841),
    (44.0, 0.20, 1, 1.017),
    (44.0, 0.40, 2, 5.202),
]


@pytest.mark.parametrize("S,sigma,T,expected", LS_TABLE1_EUROPEAN_PUT)
def test_matches_longstaff_schwartz_table1(S, sigma, T, expected):
    """Our European put must reproduce the paper's closed-form column."""
    price = bs_put(S, K=40.0, T=T, r=0.06, sigma=sigma)
    assert price == pytest.approx(expected, abs=0.001)


def test_reduces_to_base_optionmc_when_q_is_zero():
    """With q = 0 we must agree with the original OptionMC Black-Scholes."""
    from optionmc.models import OptionPricing

    base = OptionPricing(S0=100.0, E=100.0, T=1.0, rf=0.05, sigma=0.2,
                         iterations=10)
    base_call, base_put = base.bs_analytical_price()
    our_call, our_put = bs_prices(100.0, 100.0, 1.0, 0.05, 0.2, q=0.0)

    assert our_call == pytest.approx(base_call, abs=1e-12)
    assert our_put == pytest.approx(base_put, abs=1e-12)


def test_put_call_parity_with_dividend_yield():
    """C - P = S e^{-qT} - K e^{-rT} must hold exactly."""
    S0, K, T, r, sigma, q = 612.0, 600.0, 0.25, 0.043, 0.18, 0.012
    call, put = bs_prices(S0, K, T, r, sigma, q)
    lhs = call - put
    rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_dividend_yield_lowers_call_and_raises_put():
    base_call, base_put = bs_prices(100.0, 100.0, 1.0, 0.05, 0.2, q=0.0)
    div_call, div_put = bs_prices(100.0, 100.0, 1.0, 0.05, 0.2, q=0.03)
    assert div_call < base_call
    assert div_put > base_put


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        bs_call(100.0, 100.0, T=0.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        bs_put(100.0, 100.0, T=1.0, r=0.05, sigma=0.0)


def test_implied_volatility_round_trips():
    """Price at a known sigma, invert, and get the same sigma back."""
    from src.black_scholes import implied_volatility_put

    for sigma in (0.08, 0.15, 0.32, 0.75):
        price = bs_put(768.37, 749.0, 0.2, 0.0379, sigma, q=0.0121)
        recovered = implied_volatility_put(price, 768.37, 749.0, 0.2, 0.0379,
                                           q=0.0121)
        assert recovered == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("bad_price", [
    -1.0,      # negative
    0.0,       # at the zero-volatility bound: sigma is not identified
    749.0,     # above K e^{-rT}, the highest a put can ever be worth
])
def test_implied_volatility_rejects_unattainable_prices(bad_price):
    from src.black_scholes import implied_volatility_put

    with pytest.raises(ValueError):
        implied_volatility_put(bad_price, 768.37, 749.0, 0.2, 0.0379, q=0.0121)
