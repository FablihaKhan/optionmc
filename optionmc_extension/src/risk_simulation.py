"""Real-world scenarios for the SPY price at the risk horizon.

Scope sections 13 and 14. This is the one place in the project that simulates
under the REAL-WORLD measure: the drift is the historical mu estimated from SPY
log returns, never the risk-free rate.

The reason is worth stating plainly. Risk-neutral pricing answers "what is this
contract worth today", and its drift is a no-arbitrage device, not a forecast.
Value-at-Risk asks a different question -- "how badly can this position do over
the next ten trading days" -- and that depends on how the asset actually
behaves. Substituting r for mu would silently shift the whole loss
distribution, and nothing in the code would complain.

The parameter is therefore named `real_world_drift`, so a risk-neutral value
cannot be passed in without the call site saying so out loud.
"""
import numpy as np

from .gbm import simulate_terminal_prices


def horizon_in_years(horizon_days, trading_days_per_year=252):
    """Convert a horizon in trading days to a year fraction."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return horizon_days / trading_days_per_year


def simulate_horizon_scenarios(S0, real_world_drift, sigma, horizon_days,
                               n_scenarios, trading_days_per_year=252,
                               seed=None, antithetic=True, rng=None):
    """Simulate the spot price at the end of the risk horizon.

    One exact GBM step: for a horizon that only the endpoint matters for, this
    is exact rather than approximate, and far cheaper than stepping daily.

    Parameters
    ----------
    real_world_drift : float
        Historical mu. Passing r or r - q here is the mistake scope section 7
        warns about; `sanity.check_measure_separation` exists to catch it.

    Returns
    -------
    ndarray of shape (n_scenarios,)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    T = horizon_in_years(horizon_days, trading_days_per_year)
    return simulate_terminal_prices(S0, real_world_drift, sigma, T,
                                    n_scenarios, rng, antithetic)


def scenario_summary(scenarios, S0):
    """Descriptive statistics for a set of horizon spot scenarios."""
    scenarios = np.asarray(scenarios, dtype=float)
    returns = scenarios / S0 - 1.0
    return {
        "n_scenarios": int(scenarios.size),
        "mean_spot": float(scenarios.mean()),
        "median_spot": float(np.median(scenarios)),
        "min_spot": float(scenarios.min()),
        "max_spot": float(scenarios.max()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)),
        "p01_spot": float(np.percentile(scenarios, 1)),
        "p05_spot": float(np.percentile(scenarios, 5)),
        "p95_spot": float(np.percentile(scenarios, 95)),
        "p99_spot": float(np.percentile(scenarios, 99)),
    }
