"""Geometric Brownian Motion path simulation.

The base OptionMC package draws only the terminal price S_T in a single shot,
which is all a European payoff needs. Least-Squares Monte Carlo needs the whole
path, because the exercise decision is revisited at every step -- so this
module simulates full paths (scope section 23: "do not use only terminal stock
prices for LSMC").

Two measures are kept strictly separate (scope section 7):

    option pricing   -> risk-neutral measure, drift = r - q
    portfolio VaR    -> real-world measure,   drift = mu estimated from history

The drift is therefore always an explicit argument. Use `risk_neutral_drift()`
or `estimate_gbm_parameters()` to build it, so the choice is visible at the
call site and the two worlds can never be mixed up silently.
"""
import numpy as np


def risk_neutral_drift(r, q=0.0):
    """Drift for OPTION PRICING: r - q. Never use this for portfolio VaR."""
    return r - q


def simulate_gbm_paths(S0, drift, sigma, T, n_steps, n_paths, rng,
                       antithetic=False):
    """Simulate full GBM price paths.

    S_{t+dt} = S_t * exp((drift - sigma^2/2) dt + sigma sqrt(dt) Z)

    Parameters
    ----------
    S0 : float
        Initial price.
    drift : float
        Annualised drift. r - q to price, historical mu for risk.
    sigma : float
        Annualised volatility.
    T : float
        Horizon in years.
    n_steps : int
        Number of time steps; the option is exercisable at each one.
    n_paths : int
        Number of paths. Must be even when antithetic is True.
    rng : numpy.random.Generator
        Seeded generator, so every experiment is reproducible.
    antithetic : bool
        Pair each draw Z with -Z (the variance-reduction technique the base
        OptionMC package already uses for European options).

    Returns
    -------
    ndarray, shape (n_paths, n_steps + 1)
        Column 0 is S0; column n_steps is the terminal price.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic sampling needs an even n_paths")
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([half, -half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    dt = T / n_steps
    log_increments = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def simulate_terminal_prices(S0, drift, sigma, T, n_scenarios, rng,
                             antithetic=False):
    """Simulate only the price at horizon T -- one exact GBM step.

    Used for the 10-day portfolio risk scenarios, where nothing between today
    and the horizon matters. Exact for GBM, so no discretisation error.
    """
    if antithetic:
        if n_scenarios % 2 != 0:
            raise ValueError("antithetic sampling needs an even n_scenarios")
        half = rng.standard_normal(n_scenarios // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(n_scenarios)
    return S0 * np.exp((drift - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def estimate_gbm_parameters(prices, periods_per_year=252):
    """Estimate REAL-WORLD mu and sigma from a historical price series.

    Scope section 13: log returns r_t = ln(S_t / S_{t-1}), then

        mu    = mean(r_t) * 252
        sigma = std(r_t)  * sqrt(252)

    Returns
    -------
    dict with mu, sigma, mu_daily, sigma_daily, n_returns.

    Note
    ----
    mu here is the arithmetic annualisation of the mean LOG return, i.e. the
    drift term of the GBM exponent, which is exactly what `simulate_gbm_paths`
    expects as `drift`. It is a real-world quantity: never feed it to an option
    pricer (scope section 23).
    """
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if prices.size < 3:
        raise ValueError("need at least 3 prices to estimate parameters")
    log_returns = np.diff(np.log(prices))
    mu_daily = float(np.mean(log_returns))
    sigma_daily = float(np.std(log_returns, ddof=1))
    return {
        "mu": mu_daily * periods_per_year,
        "sigma": sigma_daily * np.sqrt(periods_per_year),
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "n_returns": int(log_returns.size),
    }
