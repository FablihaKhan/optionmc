"""Deterministic crash scenarios for the protective put.

Value-at-Risk and Expected Shortfall answer "how bad does it get at a given
probability". A stress test answers a different and blunter question: if the
market falls twenty percent in ten days, what happens to this position? No
probability is attached, which is the point -- the number does not depend on
whether the model believes such a move is likely.

The put is revalued at each shocked spot with the correct *remaining* maturity:
ten trading days have passed, so the contract is ten days shorter than it is
today. Valuing it at its original maturity would credit the hedge with time
value it no longer has.

Losses are measured from each portfolio's own starting value. The protected
portfolio starts out worth more, because it owns the put, and comparing it
against the unhedged starting value would quietly count the premium twice.
"""
import numpy as np

# Scope section F. Zero is included on purpose: it separates what the hedge
# costs in a flat market from what it saves in a falling one.
DEFAULT_SHOCKS = (0.0, -0.05, -0.10, -0.20, -0.30)


def shocked_spots(spot_now, shocks=DEFAULT_SHOCKS):
    """Spot after each shock, as a simple proportional move."""
    shocks = np.asarray(shocks, dtype=float)
    if np.any(shocks <= -1.0):
        raise ValueError("a shock of -100% or worse leaves no price")
    return float(spot_now) * (1.0 + shocks)


def stress_table(spot_now, put_price_now, put_value_at, shares=100,
                 contracts=1, multiplier=100, shocks=DEFAULT_SHOCKS,
                 strike=None, label="protected"):
    """Value both portfolios under each shock.

    Parameters
    ----------
    put_price_now : float
        What one put costs today, per share. This sets the protected
        portfolio's starting value, so it must be the price actually paid.
    put_value_at : callable
        Maps an array of shocked spots to the put's value per share at the
        horizon. Pass the interpolated pricing grid, or a direct pricer; the
        two are compared in the experiment rather than assumed equal.
    strike : float, optional
        Used only to check the returned values against the no-arbitrage bounds.

    Returns
    -------
    pandas.DataFrame, one row per shock.
    """
    import pandas as pd

    shocks = np.asarray(shocks, dtype=float)
    spots = shocked_spots(spot_now, shocks)
    put_values = np.asarray(put_value_at(spots), dtype=float)

    if put_values.shape != spots.shape:
        raise ValueError("put_value_at must return one value per shocked spot")
    if np.any(~np.isfinite(put_values)) or np.any(put_values < 0):
        raise ValueError("put values must be finite and non-negative")
    if strike is not None:
        intrinsic = np.maximum(strike - spots, 0.0)
        if np.any(put_values < intrinsic - 1e-6):
            raise ValueError("a put cannot be worth less than its intrinsic value")
        if np.any(put_values > strike + 1e-6):
            raise ValueError("a put cannot be worth more than its strike")

    covered = contracts * multiplier
    stock_initial = shares * float(spot_now)
    put_initial = covered * float(put_price_now)
    protected_initial = stock_initial + put_initial

    stock_value = shares * spots
    put_position = covered * put_values
    protected_value = stock_value + put_position

    stock_loss = stock_initial - stock_value
    protected_loss = protected_initial - protected_value

    return pd.DataFrame({
        "portfolio": label,
        "shock": shocks,
        "spy_price": spots,
        "put_value_per_share": put_values,
        "put_position_value": put_position,
        "stock_only_initial": stock_initial,
        "protected_initial": protected_initial,
        "stock_only_value": stock_value,
        "protected_value": protected_value,
        "stock_only_loss": stock_loss,
        "protected_loss": protected_loss,
        "stock_only_loss_percent": stock_loss / stock_initial * 100.0,
        "protected_loss_percent": protected_loss / protected_initial * 100.0,
        "hedge_benefit_dollars": stock_loss - protected_loss,
        "hedge_benefit_percent": (stock_loss / stock_initial
                                  - protected_loss / protected_initial) * 100.0,
    })


def consistency_report(table, shares=100, contracts=1, multiplier=100,
                       tolerance=1e-8):
    """Check the table against its own definitions.

    Cheap to run and worth running: an accounting slip here would move every
    stress number in the same direction and look entirely plausible.
    """
    covered = contracts * multiplier
    checks = {
        "protected value is stock plus put position": np.allclose(
            table["protected_value"],
            shares * table["spy_price"] + table["put_position_value"],
            atol=tolerance),
        "put position is the per-share value times the covered shares":
            np.allclose(table["put_position_value"],
                        covered * table["put_value_per_share"], atol=tolerance),
        "stock loss is its own start minus its own value": np.allclose(
            table["stock_only_loss"],
            table["stock_only_initial"] - table["stock_only_value"],
            atol=tolerance),
        "protected loss is its own start minus its own value": np.allclose(
            table["protected_loss"],
            table["protected_initial"] - table["protected_value"],
            atol=tolerance),
        "hedge benefit is the difference of the two losses": np.allclose(
            table["hedge_benefit_dollars"],
            table["stock_only_loss"] - table["protected_loss"], atol=tolerance),
        "the protected portfolio starts out worth more": bool(
            (table["protected_initial"] > table["stock_only_initial"]).all()),
    }
    return checks


def describe_protection(table):
    """State what the table shows, from the table rather than from belief.

    Returns a dict a caller can print. Whether the hedge helps at all, and
    where it starts helping, are read off the numbers -- a protective put loses
    money in a flat market and that has to be allowed to show.
    """
    zero = table[np.isclose(table["shock"], 0.0)]
    falls = table[table["shock"] < 0].sort_values("shock", ascending=False)

    benefit = table["hedge_benefit_dollars"].to_numpy()
    helps_anywhere = bool(np.any(benefit > 0))
    helping = table[table["hedge_benefit_dollars"] > 0]

    deepening = None
    if len(falls) >= 2:
        ordered = falls["hedge_benefit_dollars"].to_numpy()
        deepening = bool(np.all(np.diff(ordered) > 0))

    return {
        "cost_in_a_flat_market": (float(zero["hedge_benefit_dollars"].iloc[0])
                                  if len(zero) else float("nan")),
        "helps_anywhere": helps_anywhere,
        "first_shock_that_helps": (float(helping["shock"].max())
                                   if len(helping) else float("nan")),
        "benefit_grows_with_the_shock": deepening,
        "largest_benefit": float(benefit.max()),
        "largest_benefit_shock": float(
            table.loc[table["hedge_benefit_dollars"].idxmax(), "shock"]),
        "worst_protected_loss_percent": float(
            table["protected_loss_percent"].max()),
        "worst_unhedged_loss_percent": float(
            table["stock_only_loss_percent"].max()),
    }
