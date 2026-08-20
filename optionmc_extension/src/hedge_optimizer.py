"""Protective put optimizer: which real SPY put protects best for its cost?

The existing project answers "what is one American put worth, and how much tail
risk does it remove". This module turns that into a decision: several real
listed puts are priced, hedged with, and compared on protection against cost.

Three design choices are worth stating, because each one changes the answer.

**Ask, not mid.** An investor buying protection pays the ask. The mid is kept
alongside for model-versus-market comparison, but every cost figure here is
what the hedge would actually cost to put on.

**One volatility per contract, calibrated to that contract.** Each candidate's
sigma comes from inverting its own market mid against the CRR American tree.
Out-of-the-money puts trade at higher implied volatility than at-the-money ones
-- the skew is real, and pricing every strike off a single sigma would quietly
misprice the cheap wings. Because both LSMC and CRR then use the same sigma,
the difference between them measures the numerics, not the market; whether the
model tracks the market out of sample is a separate question that the
cross-section validation answers.

**One set of scenarios for every candidate.** The horizon spots are simulated
once and reused, so the difference between two strikes is the hedge rather than
Monte Carlo noise. This is common random numbers, and without it a 2% gap in
CVaR reduction could easily be sampling error.

The measure separation the scope insists on still holds: the scenarios use the
real-world drift mu, while every option value -- today's and the horizon grid's
-- uses the risk-neutral drift r - q.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .binomial import crr_american_put, implied_volatility_american_put
from .interpolation import PricingGridInterpolator
from .lsmc import price_american_put_lsmc
from .market_data import MarketDataError, select_put_contract
from .portfolio import protective_put_portfolio, unhedged_portfolio
from .pricing_grid import (build_pricing_grid, grid_matches, load_grid,
                           moneyness_grid, save_grid)
from .var_cvar import risk_measures, risk_reduction

# Scope: strikes near these fractions of spot. Not invented strikes -- the
# nearest actually listed strike to each is used.
DEFAULT_TARGET_MONEYNESS = (0.900, 0.925, 0.950, 0.975, 1.000)


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------

@dataclass
class HedgeCandidate:
    """One real listed put contract, with the quote it was selected on."""
    target_moneyness: float
    strike: float
    moneyness: float
    contract_symbol: str
    expiry: str
    days_to_expiry: int
    bid: float
    ask: float
    mid: float
    last_price: float
    quoted_iv: float            # the chain's own field; recorded, never trusted
    price_source: str           # "mid" or "last" -- reported, never silent
    volume: float
    open_interest: float
    as_of: str

    def to_row(self):
        return asdict(self)


def nearest_listed_strike(strikes, target):
    """The listed strike closest to `target`.

    Strikes are never invented: whatever the exchange lists is what an investor
    can actually buy, so the target ratios only choose among real contracts.
    """
    strikes = np.asarray(strikes, dtype=float)
    if strikes.size == 0:
        raise MarketDataError("no strikes to choose from")
    return float(strikes[np.abs(strikes - target).argmin()])


def select_candidate_puts(puts, spot, target_moneyness=DEFAULT_TARGET_MONEYNESS,
                          expiry=None, days_to_expiry=None, as_of=None):
    """Pick one real listed put per target moneyness.

    Duplicates are removed: when two target ratios land on the same listed
    strike, that contract appears once, tagged with the first target it served.

    A contract with no usable quote is skipped and returned in the rejects list
    rather than dropped silently.

    Returns
    -------
    (candidates, rejects)
        `candidates` is a list of HedgeCandidate; `rejects` is a list of dicts
        carrying the strike and the reason it could not be used.
    """
    if puts is None or len(puts) == 0:
        raise MarketDataError("empty put chain")
    if "strike" not in puts.columns:
        raise MarketDataError("put chain has no strike column")

    candidates, rejects, seen = [], [], set()

    for target in target_moneyness:
        strike = nearest_listed_strike(puts["strike"].to_numpy(), target * spot)
        if strike in seen:
            continue
        seen.add(strike)

        row = puts[puts["strike"] == strike].iloc[0]

        # Reuse the project's quote convention -- the mid/last fallback and the
        # bid/ask validation live in one place and are not reimplemented here.
        # The moneyness window is pinched around this one strike so the helper
        # resolves to exactly the contract we picked.
        band = strike / spot
        try:
            quote = select_put_contract(puts, spot, band - 1e-9, band + 1e-9)
        except MarketDataError as exc:
            rejects.append({"target_moneyness": target, "strike": strike,
                            "reason": str(exc)})
            continue

        if not (np.isfinite(quote["ask"]) and quote["ask"] > 0):
            rejects.append({"target_moneyness": target, "strike": strike,
                            "reason": "no positive ask, so the hedge has no "
                                      "acquisition cost"})
            continue

        def _field(name, default=float("nan")):
            value = row.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        candidates.append(HedgeCandidate(
            target_moneyness=float(target),
            strike=strike,
            moneyness=strike / spot,
            contract_symbol=str(row.get("contractSymbol", "")),
            expiry=str(row.get("expiry", expiry or "")),
            days_to_expiry=int(days_to_expiry) if days_to_expiry is not None else -1,
            bid=quote["bid"],
            ask=quote["ask"],
            mid=quote["market_price"],
            last_price=quote["last_price"],
            quoted_iv=quote["implied_volatility"],
            price_source=quote["price_source"],
            volume=_field("volume"),
            open_interest=_field("openInterest"),
            as_of=str(as_of or ""),
        ))

    if not candidates:
        raise MarketDataError("no candidate put had a usable quote")
    return candidates, rejects


# --------------------------------------------------------------------------
# Cost of the hedge
# --------------------------------------------------------------------------

def premium_cost(ask, contracts=1, multiplier=100):
    """What buying the protection costs, at the ask.

    Not the mid: the mid is where the market is, the ask is where a buyer
    trades.
    """
    if ask < 0:
        raise ValueError("ask cannot be negative")
    if contracts <= 0 or multiplier <= 0:
        raise ValueError("contracts and multiplier must be positive")
    return float(ask) * contracts * multiplier


def cost_percent(premium, stock_position_value):
    """Hedge premium as a percentage of the share position it protects."""
    if stock_position_value <= 0:
        raise ValueError("stock_position_value must be positive")
    return premium / stock_position_value * 100.0


# --------------------------------------------------------------------------
# Per-candidate pricing
# --------------------------------------------------------------------------

def calibrate_candidate(candidate, spot, time_to_expiry, risk_free_rate,
                        dividend_yield=0.0, n_steps=500):
    """Volatility that reproduces this contract's own market price under CRR.

    Inverting the American tree rather than Black-Scholes: a European formula
    applied to an American quote absorbs the early-exercise premium into sigma
    and biases every price built on it.
    """
    sigma = implied_volatility_american_put(
        candidate.mid, spot, candidate.strike, time_to_expiry,
        risk_free_rate, dividend_yield, n_steps=n_steps)
    return {"sigma": sigma, "calibration_price": candidate.mid,
            "calibration_source": candidate.price_source}


def price_candidate(candidate, sigma, spot, time_to_expiry, risk_free_rate,
                    dividend_yield=0.0, n_paths=10_000, n_steps=50, degree=2,
                    seed=None, binomial_steps=1000, antithetic=True):
    """LSMC and CRR prices for one candidate, and the gap between them.

    Both use the same sigma, so the gap is the numerical disagreement between
    an independent simulation method and an independent lattice method. That is
    the check the scope asks for; it is not a claim about the market.
    """
    result = price_american_put_lsmc(
        S0=spot, K=candidate.strike, T=time_to_expiry, r=risk_free_rate,
        sigma=sigma, q=dividend_yield, n_paths=n_paths, n_steps=n_steps,
        degree=degree, seed=seed, antithetic=antithetic)
    binomial = crr_american_put(spot, candidate.strike, time_to_expiry,
                                risk_free_rate, sigma, dividend_yield,
                                n_steps=binomial_steps)
    absolute = result.price - binomial
    return {
        "sigma": sigma,
        "lsmc_price": result.price,
        "lsmc_std_error": result.std_error,
        "binomial_price": binomial,
        "lsmc_minus_binomial": absolute,
        "abs_error_vs_binomial": abs(absolute),
        "rel_error_vs_binomial": abs(absolute) / binomial if binomial else float("nan"),
        "early_exercise_premium": result.early_exercise_premium,
        "early_exercise_fraction": result.early_exercise_fraction,
        "market_mid": candidate.mid,
        "market_ask": candidate.ask,
    }


def candidate_grid(candidate, sigma, spots, t_remaining, risk_free_rate,
                   dividend_yield=0.0, n_paths=200_000, n_steps=50, degree=3,
                   seed=None, cache_dir=None, rebuild=False):
    """Build (or reload) this candidate's American put grid at the horizon.

    One grid per contract, cached to disk: the grid is read once per scenario
    and there are tens of thousands of scenarios, so re-deriving it would be
    the nested Monte Carlo the scope forbids.

    Returns
    -------
    (PricingGrid, source) where source is "cached" or "built".
    """
    path = None
    if cache_dir is not None:
        cache_dir = _ensure_dir(cache_dir)
        path = cache_dir / f"grid_K{candidate.strike:g}.json"
        if not rebuild and path.exists():
            grid = load_grid(path)
            if grid_matches(grid, candidate.strike, t_remaining,
                            risk_free_rate, sigma, dividend_yield):
                return grid, "cached"

    grid = build_pricing_grid(
        spots=spots, K=candidate.strike, T_remaining=t_remaining,
        r=risk_free_rate, sigma=sigma, q=dividend_yield, n_paths=n_paths,
        n_steps=n_steps, degree=degree, seed=seed, antithetic=True)

    if path is not None:
        save_grid(grid, path)
    return grid, "built"


def _ensure_dir(path):
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Per-candidate risk
# --------------------------------------------------------------------------

def candidate_risk(candidate, grid, horizon_spots, spot, baseline,
                   shares=100, contracts=1, multiplier=100,
                   levels=(0.95, 0.99), interpolation_method="pchip"):
    """Tail risk of holding the shares plus this candidate put.

    `baseline` is the unhedged Portfolio, evaluated once on the same scenarios
    so that every candidate is measured against an identical benchmark.

    The put is valued at the horizon by interpolating this candidate's own
    pricing grid. Its cost today is the ask, so the protected portfolio starts
    out worth the shares plus what the protection actually cost.
    """
    interpolate = PricingGridInterpolator(grid, interpolation_method)
    coverage = interpolate.out_of_range(horizon_spots)
    horizon_put = interpolate(horizon_spots)

    ask = candidate.ask
    protected = protective_put_portfolio(
        shares, spot, horizon_spots, ask, horizon_put,
        contracts=contracts, multiplier=multiplier,
        name=f"SPY + put K={candidate.strike:g}")

    premium = premium_cost(ask, contracts, multiplier)
    stock_value = shares * spot

    unhedged_dollars = risk_measures(baseline.losses, levels)
    protected_dollars = risk_measures(protected.losses, levels)
    unhedged_percent = risk_measures(baseline.percent_losses, levels)
    protected_percent = risk_measures(protected.percent_losses, levels)

    row = {
        "strike": candidate.strike,
        "moneyness": candidate.moneyness,
        "ask": ask,
        "mid": candidate.mid,
        "premium_cost": premium,
        "hedge_cost_percent": cost_percent(premium, stock_value),
        "initial_value": protected.initial_value,
        "scenarios_outside_grid": coverage["total"],
        "fraction_outside_grid": coverage["fraction"],
    }

    for level in levels:
        key = f"{level:.0%}".replace("%", "")
        for stat in ("var", "cvar"):
            name = f"{stat}_{key}"
            row[f"{name}_dollars"] = protected_dollars[name]
            row[f"{name}_percent"] = protected_percent[name] * 100.0
            # Dollar basis is the ranking basis: every candidate shares the
            # same unhedged benchmark, so the comparison is like for like.
            row[f"{name}_reduction"] = risk_reduction(unhedged_dollars[name],
                                                      protected_dollars[name])
            # Percentage-of-own-value basis, reported alongside because the
            # protected portfolio starts out worth more -- it owns the put.
            row[f"{name}_reduction_pct_basis"] = risk_reduction(
                unhedged_percent[name], protected_percent[name])
            saved = unhedged_dollars[name] - protected_dollars[name]
            row[f"{name}_dollars_saved"] = saved
            row[f"{name}_saved_per_premium_dollar"] = (
                saved / premium if premium > 0 else float("nan"))

    row["worse_than_unhedged_fraction"] = float(
        (protected.percent_losses > baseline.percent_losses).mean())
    return row, protected


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def min_max_score(values, higher_is_better=True):
    """Scale to [0, 1] across the candidate set.

    When every candidate scores alike the spread is zero and there is nothing
    to distinguish; the score is then 0.5 for all of them rather than a
    division by zero. Stated here because a silent nan would propagate into the
    recommendation.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 0:
        return np.full(values.shape, 0.5)
    scaled = (values - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def pareto_mask(costs, protections):
    """Which candidates are not dominated: cheaper and stronger at once.

    Candidate i is dominated when some j costs no more and protects no less,
    and is strictly better on at least one of the two. Everything else is on
    the frontier.
    """
    costs = np.asarray(costs, dtype=float)
    protections = np.asarray(protections, dtype=float)
    if costs.shape != protections.shape:
        raise ValueError("costs and protections must have the same shape")

    efficient = np.ones(costs.shape, dtype=bool)
    for i in range(costs.size):
        cheaper_or_equal = costs <= costs[i]
        stronger_or_equal = protections >= protections[i]
        strictly_better = (costs < costs[i]) | (protections > protections[i])
        if np.any(cheaper_or_equal & stronger_or_equal & strictly_better):
            efficient[i] = False
    return efficient


def rank_candidates(frame, protection_weight=0.5, cost_weight=0.5,
                    protection_column="cvar_99_reduction",
                    cost_column="premium_cost"):
    """Score and rank the candidates, four ways, with nothing hidden.

    The balanced score is

        protection_score = minmax(protection, higher is better)
        cost_score       = minmax(cost, lower is better)
        balanced         = w_p * protection_score + w_c * cost_score

    with the weights required to sum to 1 so the score stays on [0, 1] and two
    runs with different weights remain comparable.

    Returns
    -------
    (frame, winners)
        `frame` gains the score, rank and pareto_efficient columns; `winners`
        maps each category to the winning row. No row is labelled "best" --
        each category answers a different question, and a dominated candidate
        is never called efficient.
    """
    if protection_weight < 0 or cost_weight < 0:
        raise ValueError("weights cannot be negative")
    total = protection_weight + cost_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {total}")
    if len(frame) == 0:
        raise ValueError("no candidates to rank")

    frame = frame.copy()
    frame["protection_score"] = min_max_score(frame[protection_column], True)
    frame["cost_score"] = min_max_score(frame[cost_column], False)
    frame["balanced_score"] = (protection_weight * frame["protection_score"]
                               + cost_weight * frame["cost_score"])
    frame["pareto_efficient"] = pareto_mask(frame[cost_column].to_numpy(),
                                            frame[protection_column].to_numpy())

    winners = {
        "cheapest": frame.loc[frame[cost_column].idxmin()],
        "strongest": frame.loc[frame[protection_column].idxmax()],
        "most_efficient": frame.loc[
            frame["cvar_99_saved_per_premium_dollar"].idxmax()],
        "balanced": frame.loc[frame["balanced_score"].idxmax()],
    }
    return frame, winners


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def evaluate_candidates(candidates, snapshot, horizon_spots, t_remaining,
                        grid_spots, *, shares=100, contracts=1, multiplier=100,
                        levels=(0.95, 0.99), lsmc_paths=10_000, lsmc_steps=50,
                        lsmc_degree=2, binomial_steps=1000, grid_paths=200_000,
                        grid_steps=50, grid_degree=3, seed=42,
                        interpolation_method="pchip", cache_dir=None,
                        rebuild_grids=False, iv_tree_steps=500, progress=None):
    """Calibrate, price, hedge and measure every candidate.

    `snapshot` supplies spot, risk_free_rate, dividend_yield and
    time_to_expiry; anything with those attributes works, which keeps this
    testable without a market download.

    The unhedged baseline is built once from `horizon_spots` and reused, so all
    candidates are compared on identical scenarios.

    Returns
    -------
    (rows, baseline, extras)
        `rows` is a list of dicts ready for a DataFrame; `baseline` is the
        unhedged Portfolio; `extras` carries the per-candidate grid, protected
        portfolio and grid source for callers that want to plot them.
    """
    import pandas as pd

    baseline = unhedged_portfolio(shares, snapshot.spot, horizon_spots)
    rows, extras = [], []

    for index, candidate in enumerate(candidates):
        if progress is not None:
            progress(index, candidate)

        calibration = calibrate_candidate(
            candidate, snapshot.spot, snapshot.time_to_expiry,
            snapshot.risk_free_rate, snapshot.dividend_yield,
            n_steps=iv_tree_steps)
        sigma = calibration["sigma"]

        pricing = price_candidate(
            candidate, sigma, snapshot.spot, snapshot.time_to_expiry,
            snapshot.risk_free_rate, snapshot.dividend_yield,
            n_paths=lsmc_paths, n_steps=lsmc_steps, degree=lsmc_degree,
            seed=seed + index, binomial_steps=binomial_steps)

        grid, source = candidate_grid(
            candidate, sigma, grid_spots, t_remaining,
            snapshot.risk_free_rate, snapshot.dividend_yield,
            n_paths=grid_paths, n_steps=grid_steps, degree=grid_degree,
            seed=seed, cache_dir=cache_dir, rebuild=rebuild_grids)

        risk, protected = candidate_risk(
            candidate, grid, horizon_spots, snapshot.spot, baseline,
            shares=shares, contracts=contracts, multiplier=multiplier,
            levels=levels, interpolation_method=interpolation_method)

        row = candidate.to_row()
        row.update(pricing)
        row.update(risk)
        row["grid_source"] = source
        row["calibration_source"] = calibration["calibration_source"]
        rows.append(row)
        extras.append({"candidate": candidate, "grid": grid,
                       "protected": protected, "sigma": sigma})

    return rows, baseline, extras


def baseline_risk_row(baseline, levels=(0.95, 0.99)):
    """The unhedged benchmark, in the same shape as a candidate row."""
    dollars = risk_measures(baseline.losses, levels)
    percent = risk_measures(baseline.percent_losses, levels)
    row = {"strike": float("nan"), "moneyness": float("nan"),
           "premium_cost": 0.0, "hedge_cost_percent": 0.0,
           "initial_value": baseline.initial_value}
    for level in levels:
        key = f"{level:.0%}".replace("%", "")
        for stat in ("var", "cvar"):
            name = f"{stat}_{key}"
            row[f"{name}_dollars"] = dollars[name]
            row[f"{name}_percent"] = percent[name] * 100.0
            row[f"{name}_reduction"] = 0.0
    return row
