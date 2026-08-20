"""American-put pricing grid over spot, so the risk phase needs no nested MC.

Scope section 15. Valuing the hedge in every one of 50,000 risk scenarios by
running a fresh 10,000-path LSMC inside each would be 500 million paths, which
the scope explicitly rules out. Instead the put is priced once on a grid of
spot prices at the risk horizon, and each scenario reads its price off that
grid by interpolation.

Two details make the grid usable:

Common random numbers. GBM paths scale exactly with the starting price -- with
the same shocks Z, a path started at S is just (S / S_ref) times the path
started at S_ref. So one set of unit paths is simulated once and rescaled for
every grid point. Every grid price then sees identical randomness, which
removes the point-to-point jitter that would otherwise make an interpolant
wobble between nodes. It is also much faster than simulating each point.

Remaining maturity. The grid values the option as it will be at the horizon,
so it uses T - horizon, not the full T.
"""
from dataclasses import dataclass, field

import numpy as np

from .gbm import risk_neutral_drift, simulate_gbm_paths
from .lsmc import lsmc_american_put_from_paths


@dataclass
class PricingGrid:
    """American put prices on a grid of spot values, plus how they were made."""
    spots: np.ndarray
    prices: np.ndarray
    std_errors: np.ndarray = field(repr=False)
    strike: float
    time_to_expiry: float          # remaining maturity at the horizon
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    n_paths: int
    n_steps: int
    degree: int

    @property
    def spot_range(self):
        return float(self.spots.min()), float(self.spots.max())

    def is_monotone_decreasing(self, tolerance=1e-9):
        """A put must be worth less as the underlying rises."""
        return bool(np.all(np.diff(self.prices) <= tolerance))

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "spot": self.spots,
            "moneyness": self.spots / self.strike,
            "american_put_price": self.prices,
            "std_error": self.std_errors,
            "intrinsic_value": np.maximum(self.strike - self.spots, 0.0),
        })


def build_pricing_grid(spots, K, T_remaining, r, sigma, q=0.0,
                       n_paths=50_000, n_steps=50, degree=2, seed=None,
                       antithetic=True, rng=None):
    """Price an American put by LSMC at each spot, with common random numbers.

    Parameters
    ----------
    spots : array_like
        Spot values to price at, ascending.
    T_remaining : float
        Time left on the option AT THE RISK HORIZON, in years.
    r, sigma, q : float
        Risk-neutral inputs. The drift is r - q: this is pricing, so the
        historical mu must never appear here (scope section 7).

    Returns
    -------
    PricingGrid
    """
    spots = np.asarray(spots, dtype=float)
    if spots.ndim != 1 or spots.size < 2:
        raise ValueError("spots must be a 1-D array with at least two points")
    if np.any(np.diff(spots) <= 0):
        raise ValueError("spots must be strictly increasing")
    if T_remaining <= 0:
        raise ValueError("T_remaining must be positive; the risk horizon "
                         "cannot reach past the option's expiry")

    if rng is None:
        rng = np.random.default_rng(seed)

    # One set of paths started at 1.0, reused for every grid point.
    unit_paths = simulate_gbm_paths(
        S0=1.0, drift=risk_neutral_drift(r, q), sigma=sigma, T=T_remaining,
        n_steps=n_steps, n_paths=n_paths, rng=rng, antithetic=antithetic)

    dt = T_remaining / n_steps
    prices = np.empty(spots.size)
    std_errors = np.empty(spots.size)

    for i, spot in enumerate(spots):
        result = lsmc_american_put_from_paths(spot * unit_paths, K=K, r=r,
                                              dt=dt, degree=degree)
        prices[i] = result.price
        std_errors[i] = result.std_error

    return PricingGrid(
        spots=spots, prices=prices, std_errors=std_errors, strike=K,
        time_to_expiry=T_remaining, risk_free_rate=r, dividend_yield=q,
        volatility=sigma, n_paths=n_paths, n_steps=n_steps, degree=degree)


def moneyness_grid(S0, min_moneyness, max_moneyness, n_points):
    """Spot grid spanning min..max times the current spot (scope section 15)."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if not 0 < min_moneyness < max_moneyness:
        raise ValueError("need 0 < min_moneyness < max_moneyness")
    return S0 * np.linspace(min_moneyness, max_moneyness, n_points)


def price_at_spots_directly(spots, grid, seed=None, rng=None):
    """Re-price at arbitrary spots with the SAME construction as the grid.

    Used to measure interpolation error on its own: because these prices reuse
    the grid's random-number scheme, any difference from the interpolant is the
    interpolation error rather than a fresh Monte Carlo wobble.
    """
    spots = np.asarray(spots, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)

    unit_paths = simulate_gbm_paths(
        S0=1.0,
        drift=risk_neutral_drift(grid.risk_free_rate, grid.dividend_yield),
        sigma=grid.volatility, T=grid.time_to_expiry, n_steps=grid.n_steps,
        n_paths=grid.n_paths, rng=rng, antithetic=True)

    dt = grid.time_to_expiry / grid.n_steps
    prices = np.empty(spots.size)
    for i, spot in enumerate(spots):
        prices[i] = lsmc_american_put_from_paths(
            spot * unit_paths, K=grid.strike, r=grid.risk_free_rate, dt=dt,
            degree=grid.degree).price
    return prices


def save_grid(grid, path):
    """Write a PricingGrid to JSON, values and construction settings together.

    The settings travel with the values so a later phase can verify the grid it
    loads was built for the contract it is about to use, instead of silently
    pricing a hedge with a stale curve.
    """
    import json

    payload = {
        "spots": grid.spots.tolist(),
        "prices": grid.prices.tolist(),
        "std_errors": grid.std_errors.tolist(),
        "strike": grid.strike,
        "time_to_expiry": grid.time_to_expiry,
        "risk_free_rate": grid.risk_free_rate,
        "dividend_yield": grid.dividend_yield,
        "volatility": grid.volatility,
        "n_paths": grid.n_paths,
        "n_steps": grid.n_steps,
        "degree": grid.degree,
    }
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_grid(path):
    """Read a PricingGrid back from JSON."""
    import json

    with open(str(path), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return PricingGrid(
        spots=np.asarray(payload["spots"], dtype=float),
        prices=np.asarray(payload["prices"], dtype=float),
        std_errors=np.asarray(payload["std_errors"], dtype=float),
        strike=payload["strike"],
        time_to_expiry=payload["time_to_expiry"],
        risk_free_rate=payload["risk_free_rate"],
        dividend_yield=payload["dividend_yield"],
        volatility=payload["volatility"],
        n_paths=payload["n_paths"],
        n_steps=payload["n_steps"],
        degree=payload["degree"],
    )


def grid_matches(grid, K, T_remaining, r, sigma, q, tolerance=1e-9):
    """True when a loaded grid was built for exactly this contract."""
    return (abs(grid.strike - K) < tolerance
            and abs(grid.time_to_expiry - T_remaining) < tolerance
            and abs(grid.risk_free_rate - r) < tolerance
            and abs(grid.volatility - sigma) < tolerance
            and abs(grid.dividend_yield - q) < tolerance)
