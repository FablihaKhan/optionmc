#!/usr/bin/env python
"""PHASE 7: build the American-put pricing grid and verify the interpolation.

Scope section 15. Prices the put on a grid of spot values at the risk horizon,
then verifies the interpolation at random points.

The verification separates three errors that would otherwise be confused with
one another, since only their sum is directly visible:

  (a) interpolation error   the scheme's own error, measured on a grid filled
                            with exact binomial values, so no Monte Carlo noise
                            is involved. Falls as the grid is refined.
  (b) grid node error       what the LSMC itself gets wrong at the nodes,
                            measured against the tree. Does NOT fall with more
                            paths -- it is a bias from the exercise rule.
  (c) total error           what the risk phase actually inherits.

    python experiments/interpolation_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.binomial import crr_american_put, crr_bermudan_put
from src.interpolation import (PricingGridInterpolator,
                               assess_interpolation_accuracy,
                               random_check_spots)
from src.market_data import MarketSnapshot
from src.pricing_grid import (PricingGrid, build_pricing_grid, moneyness_grid,
                              price_at_spots_directly, save_grid)
from src.sanity import Check, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_CSV = config.TABLES_DIR / "pricing_grid.csv"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
ACCURACY_CSV = config.TABLES_DIR / "interpolation_accuracy.csv"
CHECKS_CSV = config.TABLES_DIR / "interpolation_check_points.csv"

BINOMIAL_STEPS = config.BINOMIAL_N_STEPS


def main():
    if not SNAPSHOT_JSON.exists():
        print(f"No market snapshot at {SNAPSHOT_JSON}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    horizon_years = config.RISK_HORIZON_DAYS / config.TRADING_DAYS_PER_YEAR
    T_remaining = snap.time_to_expiry - horizon_years
    if T_remaining <= 0:
        print("The risk horizon reaches past the option's expiry; "
              "pick a longer-dated contract.")
        return 1

    tree = dict(K=snap.strike, T=T_remaining, r=snap.risk_free_rate,
                sigma=snap.historical_volatility, q=snap.dividend_yield)

    def binomial(spot):
        return crr_american_put(S0=spot, n_steps=BINOMIAL_STEPS, **tree)

    print("=" * 78)
    print("PHASE 7  American-put pricing grid and interpolation")
    print("=" * 78)
    print(f"  contract: SPY American put, K={snap.strike}, exp {snap.expiry}")
    print(f"  today T = {snap.time_to_expiry:.4f} yr; at the "
          f"{config.RISK_HORIZON_DAYS}-day horizon T = {T_remaining:.4f} yr")
    print(f"  grid: {config.GRID_N_POINTS} spots, "
          f"{config.GRID_MIN_MONEYNESS:.2f}*S0 .. {config.GRID_MAX_MONEYNESS:.2f}*S0"
          f"  ({config.GRID_MIN_MONEYNESS * snap.spot:.2f} .. "
          f"{config.GRID_MAX_MONEYNESS * snap.spot:.2f})")
    print(f"  LSMC per node: {config.GRID_N_PATHS:,} paths x "
          f"{config.GRID_N_STEPS} steps, degree {config.GRID_DEGREE}, "
          f"common random numbers")

    spots = moneyness_grid(snap.spot, config.GRID_MIN_MONEYNESS,
                           config.GRID_MAX_MONEYNESS, config.GRID_N_POINTS)

    start = time.perf_counter()
    grid = build_pricing_grid(
        spots=spots, K=snap.strike, T_remaining=T_remaining,
        r=snap.risk_free_rate, sigma=snap.historical_volatility,
        q=snap.dividend_yield, n_paths=config.GRID_N_PATHS,
        n_steps=config.GRID_N_STEPS, degree=config.GRID_DEGREE,
        seed=config.SEED, antithetic=True)
    grid_time = time.perf_counter() - start

    binomial_nodes = np.array([binomial(s) for s in grid.spots])
    bermudan_nodes = np.array([
        crr_bermudan_put(S0=s, n_steps=BINOMIAL_STEPS,
                         n_exercise_dates=config.GRID_N_STEPS, **tree)
        for s in grid.spots])

    frame = grid.to_frame()
    frame["binomial_price"] = binomial_nodes
    frame["bermudan_price"] = bermudan_nodes
    frame["lsmc_minus_binomial"] = grid.prices - binomial_nodes
    frame["lsmc_minus_bermudan"] = grid.prices - bermudan_nodes
    frame["discretisation_error"] = binomial_nodes - bermudan_nodes
    frame.to_csv(GRID_CSV, index=False)
    save_grid(grid, GRID_JSON)

    node_error = np.abs(grid.prices - binomial_nodes)
    mc_error = np.abs(grid.prices - bermudan_nodes)
    discr_error = np.abs(binomial_nodes - bermudan_nodes)

    print(f"\n  grid built in {grid_time:.1f} s "
          f"({grid_time / config.GRID_N_POINTS * 1000:.0f} ms per node)")

    print(f"\n  {'spot':>9} {'m=S/K':>8} {'LSMC':>10} {'binomial':>10} "
          f"{'diff':>9} {'intrinsic':>10}")
    print("  " + "-" * 60)
    step = max(1, config.GRID_N_POINTS // 10)
    for i in range(0, config.GRID_N_POINTS, step):
        print(f"  {grid.spots[i]:>9.2f} {grid.spots[i] / snap.strike:>8.4f} "
              f"{grid.prices[i]:>10.4f} {binomial_nodes[i]:>10.4f} "
              f"{grid.prices[i] - binomial_nodes[i]:>9.4f} "
              f"{max(snap.strike - grid.spots[i], 0.0):>10.4f}")

    # --- (a) interpolation error alone, on an exact grid ------------------
    rng = np.random.default_rng(config.SEED + 1)
    check_spots = random_check_spots(grid, config.INTERPOLATION_CHECK_POINTS, rng)
    check_binomial = np.array([binomial(s) for s in check_spots])

    exact_grid = PricingGrid(
        spots=grid.spots, prices=binomial_nodes,
        std_errors=np.zeros_like(binomial_nodes), strike=grid.strike,
        time_to_expiry=grid.time_to_expiry,
        risk_free_rate=grid.risk_free_rate, dividend_yield=grid.dividend_yield,
        volatility=grid.volatility, n_paths=0, n_steps=grid.n_steps,
        degree=grid.degree)
    pure = assess_interpolation_accuracy(exact_grid, check_spots, check_binomial)

    print(f"\n  (a) INTERPOLATION ERROR ALONE  "
          f"(exact grid values, {config.INTERPOLATION_CHECK_POINTS} random spots)")
    print(f"      {'method':>8} {'max abs':>11} {'mean abs':>11} {'max rel':>10}")
    for row in pure:
        print(f"      {row['method']:>8} {row['max_absolute_error']:>11.6f} "
              f"{row['mean_absolute_error']:>11.6f} "
              f"{row['max_relative_error']:>10.4%}")

    # --- (b) grid node error ----------------------------------------------
    print("\n  (b) GRID NODE ERROR  (LSMC vs tree, at the nodes)")
    print(f"      LSMC vs Bermudan on the same {config.GRID_N_STEPS}-date grid"
          f"  max {mc_error.max():.5f}   mean {mc_error.mean():.5f}"
          "   <- Monte Carlo")
    print(f"      Bermudan vs continuous American                    "
          f"  max {discr_error.max():.5f}   mean {discr_error.mean():.5f}"
          "   <- discretisation")
    print(f"      LSMC vs continuous American (their sum)            "
          f"  max {node_error.max():.5f}   mean {node_error.mean():.5f}")

    # --- (c) total error at the check points ------------------------------
    direct_lsmc = price_at_spots_directly(check_spots, grid, seed=config.SEED)
    total = assess_interpolation_accuracy(grid, check_spots, check_binomial)

    print("\n  (c) TOTAL ERROR  (LSMC grid, interpolated, vs the tree)")
    print(f"      {'method':>8} {'max abs':>11} {'mean abs':>11} {'max rel':>10}")
    for row in total:
        print(f"      {row['method']:>8} {row['max_absolute_error']:>11.6f} "
              f"{row['mean_absolute_error']:>11.6f} "
              f"{row['max_relative_error']:>10.4%}")

    accuracy = pd.concat([
        pd.DataFrame(pure).assign(stage="a_interpolation_only"),
        pd.DataFrame(total).assign(stage="c_total_vs_binomial"),
    ])
    accuracy.to_csv(ACCURACY_CSV, index=False)

    chosen = PricingGridInterpolator(grid, config.INTERPOLATION_METHOD)
    pd.DataFrame({
        "spot": check_spots,
        "moneyness": check_spots / snap.strike,
        "interpolated": chosen(check_spots),
        "direct_lsmc": direct_lsmc,
        "binomial": check_binomial,
        "interp_minus_lsmc": chosen(check_spots) - direct_lsmc,
        "interp_minus_binomial": chosen(check_spots) - check_binomial,
    }).to_csv(CHECKS_CSV, index=False)

    # --- sanity checks -----------------------------------------------------
    dense = np.linspace(*grid.spot_range, 4001)
    dense_values = chosen(dense)
    pure_chosen = next(r for r in pure
                       if r["method"] == config.INTERPOLATION_METHOD)
    total_chosen = next(r for r in total
                        if r["method"] == config.INTERPOLATION_METHOD)

    # Relative error is reported as a profile rather than tested against one
    # threshold. Deep out of the money the true price is a fraction of a cent,
    # so a relative test there measures division by almost zero: the same
    # 0.0008 absolute error reads as 4.8% at a node worth 1.7 cents and as
    # 0.02% at a node worth 4 dollars. What the portfolio actually feels is
    # the absolute error in dollars, so that is what the checks below use.
    contracts = config.CONTRACT_MULTIPLIER
    print("\n  relative node error by price floor (a profile, not a test):")
    for floor in (0.01, 0.10, 1.00, 2.00):
        material = binomial_nodes >= floor
        if not material.any():
            continue
        rel = node_error[material] / binomial_nodes[material]
        print(f"      put worth >= ${floor:5.2f}: {int(material.sum()):3d} nodes, "
              f"max rel {float(rel.max()):>7.4%}, "
              f"max abs {float(node_error[material].max()):.5f}")

    print("\n  sanity checks:")
    checks = [
        Check("grid price falls as spot rises",
              grid.is_monotone_decreasing(),
              f"{grid.prices[0]:.4f} down to {grid.prices[-1]:.4f}"),
        Check("grid price is at least intrinsic everywhere",
              bool(np.all(grid.prices
                          >= np.maximum(snap.strike - grid.spots, 0.0) - 1e-9)),
              f"min slack "
              f"{float(np.min(grid.prices - np.maximum(snap.strike - grid.spots, 0.0))):.4f}"),
        Check("grid node error under 10 cents per share everywhere",
              bool(node_error.max() < 0.10),
              f"max {node_error.max():.5f}"),
        Check(f"worst-case grid error under $10 on the {contracts}-share position",
              bool(contracts * node_error.max() < 10.0),
              f"${contracts * node_error.max():.2f} worst, "
              f"${contracts * node_error.mean():.2f} mean"),
        Check(f"{config.INTERPOLATION_METHOD} interpolation error under 2 cents",
              pure_chosen["max_absolute_error"] < 0.02,
              f"max {pure_chosen['max_absolute_error']:.6f}"),
        Check("interpolation is not the dominant error",
              pure_chosen["max_absolute_error"] < mc_error.max(),
              f"interpolation {pure_chosen['max_absolute_error']:.5f} "
              f"< Monte Carlo {mc_error.max():.5f}"),
        Check("interpolant stays monotone between the nodes",
              bool(np.all(np.diff(dense_values) <= 1e-9)),
              f"checked at {dense.size} points"),
        Check("interpolant never leaves the no-arbitrage box",
              bool(np.all(dense_values
                          >= np.maximum(snap.strike - dense, 0.0) - 1e-9)
                   and np.all(dense_values <= snap.strike + 1e-9)),
              "intrinsic <= price <= K"),
    ]
    all_passed = report(checks)

    print("\n  What this costs the risk numbers:")
    print(f"   * Worst case the hedge is mispriced by "
          f"{total_chosen['max_absolute_error']:.4f} per share, so "
          f"{contracts * total_chosen['max_absolute_error']:,.2f} dollars on the")
    print(f"     {contracts}-share position; on average "
          f"{contracts * total_chosen['mean_absolute_error']:,.2f} dollars.")
    print(f"   * The dominant term is the LSMC bias at the nodes "
          f"({mc_error.max():.4f}), not the")
    print(f"     interpolation ({pure_chosen['max_absolute_error']:.4f}). More "
          "grid points would not help; a")
    print("     better regression basis would, which is why the grid uses "
          f"degree {config.GRID_DEGREE}.")
    print(f"\n  Nested Monte Carlo avoided: valuing "
          f"{config.N_RISK_SCENARIOS:,} scenarios directly would need")
    print(f"  {config.N_RISK_SCENARIOS:,} x {config.GRID_N_PATHS:,} = "
          f"{config.N_RISK_SCENARIOS * config.GRID_N_PATHS / 1e9:,.0f} billion paths. "
          f"The grid took {grid_time:.1f} s.")
    print(f"\n  tables saved to {GRID_CSV.name}, {ACCURACY_CSV.name}, "
          f"{CHECKS_CSV.name}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
