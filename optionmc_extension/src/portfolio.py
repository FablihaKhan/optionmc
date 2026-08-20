"""The two portfolios and their loss distributions (scope sections 12 and 16).

    Portfolio A, unhedged:   V_0 = 100 S_0
                             V_h = 100 S_h

    Portfolio B, protected:  V_0 = 100 S_0 + 100 P_0
                             V_h = 100 S_h + 100 P_h

One option contract covers 100 shares, which is why the same multiplier appears
on both legs.

Losses are measured as L = V_0 - V_h, so a positive number is money lost. Both
a dollar loss and a percentage loss are reported, and the percentage one is the
comparison that is actually fair: the protected portfolio starts out worth more
because the put it holds is an asset, so equal dollar losses do not mean equal
damage.

Nothing here assumes the hedge helps. Whether it does is measured downstream in
var_cvar, from these numbers (scope section 23).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Portfolio:
    """One portfolio's initial value and its distribution of outcomes."""
    name: str
    initial_value: float
    horizon_values: np.ndarray = field(repr=False)
    losses: np.ndarray = field(repr=False)
    percent_losses: np.ndarray = field(repr=False)
    components: dict = field(default_factory=dict, repr=False)

    def summary(self):
        """Descriptive statistics of the loss distribution."""
        return {
            "portfolio": self.name,
            "initial_value": self.initial_value,
            "n_scenarios": int(self.losses.size),
            "mean_horizon_value": float(self.horizon_values.mean()),
            "mean_loss": float(self.losses.mean()),
            "std_loss": float(self.losses.std(ddof=1)),
            "min_loss": float(self.losses.min()),
            "max_loss": float(self.losses.max()),
            "mean_percent_loss": float(self.percent_losses.mean()),
            "std_percent_loss": float(self.percent_losses.std(ddof=1)),
            "worst_percent_loss": float(self.percent_losses.max()),
            "probability_of_loss": float((self.losses > 0).mean()),
        }


def _build(name, initial_value, horizon_values, components):
    horizon_values = np.asarray(horizon_values, dtype=float)
    losses = initial_value - horizon_values
    return Portfolio(
        name=name,
        initial_value=float(initial_value),
        horizon_values=horizon_values,
        losses=losses,
        percent_losses=losses / initial_value,
        components=components,
    )


def unhedged_portfolio(shares, spot_now, horizon_spots, name="SPY only"):
    """Portfolio A: shares of SPY and nothing else."""
    if shares <= 0:
        raise ValueError("shares must be positive")
    horizon_spots = np.asarray(horizon_spots, dtype=float)

    initial_value = shares * spot_now
    horizon_values = shares * horizon_spots
    return _build(name, initial_value, horizon_values,
                  {"stock_now": initial_value, "put_now": 0.0})


def protective_put_portfolio(shares, spot_now, horizon_spots, put_price_now,
                             horizon_put_prices, contracts=1, multiplier=100,
                             name="SPY + put"):
    """Portfolio B: the same shares plus a long American put.

    Parameters
    ----------
    put_price_now : float
        Value of one put today, per share.
    horizon_put_prices : array_like
        Value of one put per share in each scenario at the horizon -- read off
        the interpolated pricing grid, which is what avoids a nested Monte
        Carlo (scope section 15).
    contracts, multiplier : int
        Number of contracts and shares per contract; 1 x 100 by default, which
        is what makes the put cover the whole 100-share position.
    """
    if shares <= 0:
        raise ValueError("shares must be positive")
    if contracts <= 0 or multiplier <= 0:
        raise ValueError("contracts and multiplier must be positive")
    if put_price_now < 0:
        raise ValueError("put_price_now cannot be negative")

    horizon_spots = np.asarray(horizon_spots, dtype=float)
    horizon_put_prices = np.asarray(horizon_put_prices, dtype=float)
    if horizon_put_prices.shape != horizon_spots.shape:
        raise ValueError("horizon_put_prices must match horizon_spots in shape")
    if np.any(horizon_put_prices < 0):
        raise ValueError("horizon put prices cannot be negative")

    covered = contracts * multiplier
    stock_now = shares * spot_now
    put_now = covered * put_price_now

    initial_value = stock_now + put_now
    horizon_values = shares * horizon_spots + covered * horizon_put_prices
    return _build(name, initial_value, horizon_values,
                  {"stock_now": stock_now, "put_now": put_now,
                   "covered_shares": covered})


def hedge_coverage(shares, contracts, multiplier=100):
    """Fraction of the share position the puts actually cover.

    Reported rather than assumed: the scope's 100 shares and one contract give
    exactly 1.0, and any other combination is a partial hedge whose risk
    numbers must be read in that light.
    """
    return contracts * multiplier / shares


def compare(portfolio_a, portfolio_b):
    """Side-by-side loss statistics for two portfolios."""
    import pandas as pd
    return pd.DataFrame([portfolio_a.summary(), portfolio_b.summary()])
