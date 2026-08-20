#!/usr/bin/env python
"""ADVANCED PHASE 3b: deterministic crash scenarios for the protective put.

VaR and CVaR say how bad things get at a stated probability. This asks the
blunter question a person actually asks: if SPY falls twenty percent in ten
days, what happens to my position? No probability is attached, so the answer
does not depend on whether the model thinks such a fall is likely -- which is
exactly the point, since the bootstrap and GBM disagree about that.

    python experiments/stress_test_experiment.py

The put is revalued at the correct REMAINING maturity: ten trading days have
passed, so the contract is ten days shorter than it is today. Valuing it at its
original maturity would credit the hedge with time value it no longer has.

Each portfolio's loss is measured from its own starting value. The protected
one starts out worth more because it owns the put; measuring both from the
unhedged start would count the premium twice.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import plots, plots_risk_models
from src.binomial import crr_american_put
from src.interpolation import PricingGridInterpolator
from src.lsmc import price_american_put_lsmc
from src.market_data import MarketSnapshot
from src.pricing_grid import grid_matches, load_grid
from src.risk_simulation import horizon_in_years
from src.sanity import Check, report
from src.stress_testing import (DEFAULT_SHOCKS, consistency_report,
                                describe_protection, shocked_spots,
                                stress_table)

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
HEDGE_GRIDS = config.DATA_DIR / "hedge_grids"

RESULTS_CSV = config.TABLES_DIR / "stress_test_results.csv"
PRICER_CSV = config.TABLES_DIR / "stress_test_pricer_agreement.csv"


def main():
    if not SNAPSHOT_JSON.exists() or not GRID_JSON.exists():
        print("Run experiments/portfolio_risk.py first.")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    horizon_years = horizon_in_years(config.RISK_HORIZON_DAYS,
                                     config.TRADING_DAYS_PER_YEAR)
    t_remaining = snap.time_to_expiry - horizon_years

    print("=" * 78)
    print("ADVANCED PHASE 3b  Deterministic stress tests")
    print("=" * 78)
    print(f"  SPY {snap.spot:.2f}, {config.RISK_HORIZON_DAYS}-day horizon, "
          f"put has {t_remaining:.4f} yr left at the horizon "
          f"(from {snap.time_to_expiry:.4f} today)")

    grid = load_grid(GRID_JSON)
    if not grid_matches(grid, snap.strike, t_remaining, snap.risk_free_rate,
                        snap.historical_volatility, snap.dividend_yield):
        print("  The cached pricing grid does not match the snapshot.")
        return 1

    today = price_american_put_lsmc(
        S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
        r=snap.risk_free_rate, sigma=snap.historical_volatility,
        q=snap.dividend_yield, n_paths=config.GRID_N_PATHS,
        n_steps=config.GRID_N_STEPS, degree=config.GRID_DEGREE,
        seed=config.SEED, antithetic=True)

    hedges = [{
        "name": f"SPY + put K={snap.strike:g}",
        "strike": snap.strike,
        "grid": grid,
        "sigma": snap.historical_volatility,
        "cost_per_share": today.price,
        "cost_basis": "LSMC model price",
    }]
    efficient = load_efficient_hedge(t_remaining)
    if efficient is not None:
        hedges.append(efficient)

    spots = shocked_spots(snap.spot, DEFAULT_SHOCKS)
    lo, hi = grid.spot_range
    inside = bool(np.all((spots >= lo) & (spots <= hi)))
    print(f"  shocked spots {spots.min():.2f} .. {spots.max():.2f} against a "
          f"grid of [{lo:.2f}, {hi:.2f}]: "
          f"{'all inside' if inside else 'SOME OUTSIDE'}")

    # --- do the two pricers agree at the shocked spots? --------------------
    agreement_rows = []
    for hedge in hedges:
        interpolate = PricingGridInterpolator(hedge["grid"],
                                              config.INTERPOLATION_METHOD)
        interpolated = interpolate(spots)
        direct = np.array([
            crr_american_put(float(s), hedge["strike"], t_remaining,
                             snap.risk_free_rate, hedge["sigma"],
                             snap.dividend_yield,
                             n_steps=config.BINOMIAL_N_STEPS)
            for s in spots])
        for shock, spot, a, b in zip(DEFAULT_SHOCKS, spots, interpolated, direct):
            agreement_rows.append({
                "hedge": hedge["name"], "shock": shock, "spy_price": spot,
                "grid_interpolated": a, "crr_direct": b,
                "difference": a - b,
            })
        hedge["interpolate"] = interpolate
        hedge["direct"] = direct

    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(PRICER_CSV, index=False)

    print("\n  the horizon put value, two independent ways:")
    head = (f"    {'hedge':<22} {'shock':>7} {'SPY':>9} {'grid':>9} "
            f"{'CRR direct':>11} {'difference':>11}")
    print(head)
    print("    " + "-" * (len(head) - 5))
    for _, r in agreement.iterrows():
        print(f"    {r['hedge']:<22} {r['shock']:>+7.0%} {r['spy_price']:>9.2f} "
              f"{r['grid_interpolated']:>9.4f} {r['crr_direct']:>11.4f} "
              f"{r['difference']:>+11.4f}")
    worst_gap = float(agreement["difference"].abs().max())
    meaningful = agreement[agreement["crr_direct"] >= 1.0]
    worst_relative = float((meaningful["difference"].abs()
                            / meaningful["crr_direct"]).max())
    print(f"\n   Largest disagreement {worst_gap:.4f} per share "
          f"(${worst_gap * config.CONTRACT_MULTIPLIER:.2f} per contract), "
          f"{worst_relative:.3%} of the")
    print("   put's own value. The interpolated grid and a 2,000-step tree are")
    print("   independent routes to the same number -- one a simulation read")
    print("   through a spline, the other a lattice -- so agreement here is a")
    print("   real check on both. The gap is the LSMC's node noise, which the")
    print("   original interpolation study already measured at 0.057 against")
    print("   the tree; an absolute cent was never the right bar for it.")

    # --- the stress table ---------------------------------------------------
    tables = []
    for hedge in hedges:
        table = stress_table(
            spot_now=snap.spot, put_price_now=hedge["cost_per_share"],
            put_value_at=hedge["interpolate"], shares=config.SHARES,
            contracts=1, multiplier=config.CONTRACT_MULTIPLIER,
            shocks=DEFAULT_SHOCKS, strike=hedge["strike"],
            label=hedge["name"])
        table["put_cost_basis"] = hedge["cost_basis"]
        table["strike"] = hedge["strike"]
        tables.append(table)
        hedge["table"] = table

    results = pd.concat(tables, ignore_index=True)
    results.to_csv(RESULTS_CSV, index=False)

    for hedge in hedges:
        table = hedge["table"]
        print(f"\n  {hedge['name']}  "
              f"(premium {hedge['cost_per_share'] * config.CONTRACT_MULTIPLIER:,.2f}, "
              f"{hedge['cost_basis']})")
        head = (f"    {'shock':>7} {'SPY':>9} {'put/share':>10} "
                f"{'stock loss':>12} {'protected':>12} {'benefit $':>11} "
                f"{'stock %':>9} {'prot %':>9}")
        print(head)
        print("    " + "-" * (len(head) - 5))
        for _, r in table.iterrows():
            print(f"    {r['shock']:>+7.0%} {r['spy_price']:>9.2f} "
                  f"{r['put_value_per_share']:>10.4f} "
                  f"{r['stock_only_loss']:>12,.2f} "
                  f"{r['protected_loss']:>12,.2f} "
                  f"{r['hedge_benefit_dollars']:>+11,.2f} "
                  f"{r['stock_only_loss_percent']:>8.2f}% "
                  f"{r['protected_loss_percent']:>8.2f}%")

    # --- what it shows, read off the table ---------------------------------
    primary = hedges[0]["table"]
    story = describe_protection(primary)
    print(f"\n  what the numbers say, for {hedges[0]['name']}:")
    print(f"    In a flat market the hedge costs "
          f"{abs(story['cost_in_a_flat_market']):,.2f} dollars of portfolio "
          f"value: that is the")
    print(f"    premium decaying, and it is what protection costs when it is "
          f"not needed.")
    if story["helps_anywhere"]:
        print(f"    It first pays for itself at a "
              f"{story['first_shock_that_helps']:+.0%} shock, and the benefit "
              f"{'grows' if story['benefit_grows_with_the_shock'] else 'moves'} "
              f"with every")
        print(f"    further leg down, reaching "
              f"{story['largest_benefit']:,.2f} dollars at "
              f"{story['largest_benefit_shock']:+.0%}.")
    else:
        print("    Across these shocks the hedge never recovers its premium.")
    print(f"    At the deepest shock the unhedged position is down "
          f"{story['worst_unhedged_loss_percent']:.2f}% while the protected")
    print(f"    one is down {story['worst_protected_loss_percent']:.2f}%. The "
          f"put does not remove the loss; it caps how fast it grows.")

    # --- sanity checks ------------------------------------------------------
    print("\n  sanity checks:")
    checks = [Check("shocked spots stay inside the pricing grid", inside,
                    f"{spots.min():.2f} .. {spots.max():.2f} in "
                    f"[{lo:.2f}, {hi:.2f}]"),
              Check("the grid and a direct tree agree to 1% of the put's value",
                    worst_relative < 0.01,
                    f"worst {worst_relative:.4%} across {len(meaningful)} "
                    f"points where the put is worth at least $1"),
              Check("the pricer disagreement cannot move a stress result",
                    worst_gap * config.CONTRACT_MULTIPLIER
                    < 0.01 * float(primary_benefit(hedges)),
                    f"${worst_gap * config.CONTRACT_MULTIPLIER:.2f} against a "
                    f"smallest hedge benefit of "
                    f"${float(primary_benefit(hedges)):,.2f}")]

    for hedge in hedges:
        table = hedge["table"]
        for description, passed in consistency_report(
                table, config.SHARES, 1, config.CONTRACT_MULTIPLIER).items():
            checks.append(Check(f"{hedge['name']}: {description}", bool(passed),
                                ""))
        intrinsic = np.maximum(hedge["strike"] - table["spy_price"], 0.0)
        checks.append(Check(
            f"{hedge['name']}: the put is never below intrinsic value",
            bool(np.all(table["put_value_per_share"] >= intrinsic - 1e-6)),
            f"deepest shock intrinsic {intrinsic.max():.2f}, "
            f"value {table['put_value_per_share'].max():.2f}"))
        checks.append(Check(
            f"{hedge['name']}: the put gains value as the market falls",
            bool(table.sort_values("shock")["put_value_per_share"]
                 .is_monotonic_decreasing),
            "a lower spot means a more valuable put"))
        checks.append(Check(
            f"{hedge['name']}: the protected loss never exceeds the unhedged "
            f"one in a crash",
            bool(np.all(table[table["shock"] <= -0.05]["protected_loss"]
                        < table[table["shock"] <= -0.05]["stock_only_loss"])),
            "checked at every shock of -5% or worse"))

    all_passed = report(checks)

    # --- figure -------------------------------------------------------------
    plots.apply_style()
    fig, _ = plots_risk_models.plot_stress_test(
        primary["shock"], primary["stock_only_loss_percent"],
        primary["protected_loss_percent"], primary["put_value_per_share"],
        title=(f"A crash, ten days out: SPY only against SPY + put "
               f"K={snap.strike:g}"))
    plots.save(fig, config.FIGURES_DIR, "27_stress_test")

    print(f"\n  tables  {RESULTS_CSV.name}, {PRICER_CSV.name}")
    print("  figures 27_stress_test")
    return 0 if all_passed else 1


def primary_benefit(hedges):
    """The smallest positive hedge benefit across the stressed scenarios.

    The yardstick a pricing disagreement has to be small against: if it cannot
    shift even the weakest stress result, it cannot shift the conclusion.
    """
    table = hedges[0]["table"]
    positive = table[table["hedge_benefit_dollars"] > 0]["hedge_benefit_dollars"]
    return positive.min() if len(positive) else float("inf")


def load_efficient_hedge(t_remaining):
    """The optimizer's most efficient hedge, if phase 1 has been run."""
    rankings = config.TABLES_DIR / "hedge_optimizer_rankings.csv"
    if not rankings.exists():
        return None
    row = pd.read_csv(rankings).set_index("category").loc["most_efficient"]
    strike = float(row["strike"])
    path = HEDGE_GRIDS / f"grid_K{strike:g}.json"
    if not path.exists():
        return None
    grid = load_grid(path)
    if abs(grid.time_to_expiry - t_remaining) > 1e-9:
        return None
    return {
        "name": f"SPY + put K={strike:g}",
        "strike": strike,
        "grid": grid,
        "sigma": grid.volatility,
        "cost_per_share": float(row["ask"]),
        "cost_basis": "market ask",
    }


if __name__ == "__main__":
    sys.exit(main())
