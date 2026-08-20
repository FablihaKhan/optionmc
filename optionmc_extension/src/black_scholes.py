"""Black-Scholes analytical prices, with a continuous dividend yield.

The base OptionMC package already implements Black-Scholes, but only for a
non-dividend-paying underlying (`OptionPricing.bs_analytical_price`). SPY does
pay dividends, and the project scope requires the risk-neutral drift to be
r - q, so this module adds the q term. The base package is left untouched.

Black-Scholes is used here ONLY as the European benchmark. It is never used as
a benchmark for the American put -- that role belongs to the binomial tree
(scope section 9).
"""
import numpy as np
from scipy.stats import norm


def d1_d2(S0, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes d1 and d2 terms."""
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_call(S0, K, T, r, sigma, q=0.0):
    """European call price under Black-Scholes with dividend yield q."""
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0, K, T, r, sigma, q=0.0):
    """European put price under Black-Scholes with dividend yield q."""
    d1, d2 = d1_d2(S0, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def bs_prices(S0, K, T, r, sigma, q=0.0):
    """Return (call, put) together."""
    return bs_call(S0, K, T, r, sigma, q), bs_put(S0, K, T, r, sigma, q)


def implied_volatility_put(price, S0, K, T, r, q=0.0, lo=1e-6, hi=5.0):
    """Invert the European put formula for sigma.

    The volatility field published in a downloaded option chain is often stale
    and inconsistent with the quoted bid/ask, so this project computes its own
    implied volatility from the observed price rather than trusting that field.

    Raises
    ------
    ValueError
        If no volatility in [lo, hi] reproduces `price`, which happens when the
        quote violates the no-arbitrage bounds.
    """
    from scipy.optimize import brentq

    def objective(sigma):
        return bs_put(S0, K, T, r, sigma, q) - price

    # A price at or below the zero-volatility bound does not identify a
    # volatility at all, so it is rejected rather than returned as sigma = 0.
    if objective(lo) >= 0.0 or objective(hi) <= 0.0:
        raise ValueError(
            f"put price {price} is not attainable for sigma in [{lo}, {hi}]"
        )
    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12))
