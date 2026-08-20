"""Horizon scenarios drawn from observed SPY returns rather than from GBM.

The project's risk engine assumes geometric Brownian motion, which means daily
log returns are normal. Real SPY returns are not: over the cached five years
their excess kurtosis is close to eight, so days far out in the tail happen far
more often than a normal distribution allows. A protective put pays off exactly
in that tail, so a risk model that understates it also understates the hedge.

This module adds a second engine that makes no distributional assumption at
all. For each scenario it draws `horizon_days` observed daily log returns with
replacement, sums them, and exponentiates. Nothing is fitted -- no mean, no
variance, no normal -- so whatever shape the historical sample has is carried
straight into the horizon distribution. GBM is not replaced; the two are run
side by side and compared.

**No risk-free rate reaches this module.** There is no `r`, no `q` and no drift
parameter anywhere in its signatures, because the drift here is whatever the
sampled history contains. Option *values* at the horizon are still risk-neutral
and still come from the pricing grid; that separation is the one the scope
warns about most loudly, and here it is enforced by the interface rather than
by a comment.

One limitation worth stating: drawing days independently discards volatility
clustering, so a real ten-day crash -- several bad days in a row -- is less
likely under this bootstrap than in the market. The fat tails of single days
survive; the tendency of bad days to arrive together does not.
"""
import numpy as np


def daily_log_returns(prices):
    """Log returns r_t = log(S_t / S_{t-1}) from a price series."""
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 2:
        raise ValueError("need at least two prices to form a return")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive to take logs")
    return np.diff(np.log(prices))


def bootstrap_horizon_prices(S0, log_returns, horizon_days, n_scenarios,
                             rng=None, seed=None):
    """Terminal prices from resampled historical returns.

    Each scenario draws `horizon_days` observed daily log returns with
    replacement, sums them, and applies the total to the current spot.

    Deliberately takes no drift, rate or volatility argument. Every one of
    those would be a parameter fitted to the data, and this estimator is meant
    to use the data itself.

    Parameters
    ----------
    S0 : float
        Spot today.
    log_returns : array_like
        Observed daily log returns to resample from.
    horizon_days : int
        Trading days in the risk horizon.

    Returns
    -------
    ndarray of shape (n_scenarios,)
    """
    log_returns = np.asarray(log_returns, dtype=float)
    log_returns = log_returns[np.isfinite(log_returns)]
    if log_returns.size == 0:
        raise ValueError("no usable historical returns to resample")
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")

    if rng is None:
        rng = np.random.default_rng(seed)

    draws = rng.integers(0, log_returns.size, (int(n_scenarios),
                                               int(horizon_days)))
    return S0 * np.exp(log_returns[draws].sum(axis=1))


def bootstrap_horizon_returns(log_returns, horizon_days, n_scenarios, rng=None,
                              seed=None):
    """The summed horizon log returns themselves, for diagnostics."""
    log_returns = np.asarray(log_returns, dtype=float)
    log_returns = log_returns[np.isfinite(log_returns)]
    if rng is None:
        rng = np.random.default_rng(seed)
    draws = rng.integers(0, log_returns.size, (int(n_scenarios),
                                               int(horizon_days)))
    return log_returns[draws].sum(axis=1)


def return_distribution_summary(log_returns, label=""):
    """Shape statistics of a return sample, including the tail behaviour."""
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 4:
        raise ValueError("need at least four observations")

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    centred = (values - mean) / std
    return {
        "label": label,
        "n": int(n),
        "mean": mean,
        "std": std,
        "skewness": float(np.mean(centred ** 3)),
        "excess_kurtosis": float(np.mean(centred ** 4) - 3.0),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def quantile_comparison(empirical, simulated, probabilities=None):
    """Matched quantiles of two samples, for a QQ comparison.

    A normal fitted to a fat-tailed sample matches in the middle and misses at
    the ends; putting the quantiles side by side is what makes that visible
    rather than a claim.
    """
    if probabilities is None:
        probabilities = np.concatenate([
            np.array([0.001, 0.005]),
            np.linspace(0.01, 0.99, 99),
            np.array([0.995, 0.999]),
        ])
    probabilities = np.asarray(probabilities, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    return {
        "probability": probabilities,
        "empirical": np.quantile(empirical, probabilities),
        "simulated": np.quantile(simulated, probabilities),
    }


def tail_exceedance(sample, thresholds):
    """How often a sample falls below each threshold.

    Used to compare how many ten-day outcomes each engine puts past a given
    loss, which is the question a tail-risk model is actually asked.
    """
    sample = np.asarray(sample, dtype=float)
    return {float(t): float((sample <= t).mean()) for t in thresholds}
