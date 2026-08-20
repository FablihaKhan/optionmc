#!/usr/bin/env python
"""ADVANCED PHASE 1: which real SPY put protects best for its cost?

Takes the single protective-put result the project already has and turns it
into a decision. Five real listed SPY puts -- the nearest strikes to 90.0,
92.5, 95.0, 97.5 and 100.0 percent of spot -- are each calibrated to their own
market quote, priced by LSMC and by CRR, used to hedge the 100-share position,
and compared on how much tail risk they remove against what they cost.

    python experiments/hedge_optimizer_experiment.py
    python experiments/hedge_optimizer_experiment.py --rebuild-grids
    python experiments/hedge_optimizer_experiment.py --protection-weight 0.7

Runs entirely from cached data: the option chain already on disk carries every
strike needed. The first run builds one pricing grid per candidate and caches
it, so later runs are fast.

No hedge is declared best. Four categories are reported, each answering a
different question, and the scoring rule is printed rather than hidden.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import plots, plots_hedge
from src.hedge_optimizer import (baseline_risk_row, evaluate_candidates,
                                 rank_candidates, select_candidate_puts)
from src.interpolation import assess_interpolation_accuracy, random_check_spots
from src.market_data import MarketSnapshot
from src.pricing_grid import moneyness_grid, price_at_spots_directly
from src.risk_simulation import horizon_in_years, simulate_horizon_scenarios
from src.sanity import Check, check_measure_separation, check_risk_measures, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
CHAIN_CSV = config.SPY_OPTION_CSV
GRID_CACHE = config.DATA_DIR / "hedge_grids"

CANDIDATES_CSV = config.TABLES_DIR / "hedge_optimizer_candidates.csv"
RANKINGS_CSV = config.TABLES_DIR / "hedge_optimizer_rankings.csv"
FRONTIER_CSV = config.TABLES_DIR / "protection_cost_frontier.csv"
INTERP_CSV = config.TABLES_DIR / "hedge_optimizer_interpolation.csv"

LEVELS = tuple(config.CONFIDENCE_LEVELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-grids", action="store_true",
                        help="rebuild every candidate pricing grid")
    parser.add_argument("--protection-weight", type=float, default=0.5,
                        help="weight on protection in the balanced score")
    parser.add_argument("--cost-weight", type=float, default=0.5,
                        help="weight on cost in the balanced score")
    parser.add_argument("--skip-interpolation-check", action="store_true",
                        help="skip the direct-pricing interpolation audit")
    args = parser.parse_args()

    if not SNAPSHOT_JSON.exists() or not CHAIN_CSV.exists():
        print(f"Missing {SNAPSHOT_JSON} or {CHAIN_CSV}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1

    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)
    chain = pd.read_csv(CHAIN_CSV)

    horizon_years = horizon_in_years(config.RISK_HORIZON_DAYS,
                                     config.TRADING_DAYS_PER_YEAR)
    t_remaining = snap.time_to_expiry - horizon_years
    if t_remaining <= 0:
        print("The risk horizon reaches past expiry.")
        return 1

    print("=" * 78)
    print("ADVANCED PHASE 1  Protective put optimizer")
    print("=" * 78)
    print(f"  {snap.ticker} spot {snap.spot:.2f} as of {snap.as_of}, "
          f"expiry {snap.expiry} ({snap.days_to_expiry} days)")
    print(f"  chain on disk: {len(chain)} contracts, "
          f"strikes {chain['strike'].min():g} to {chain['strike'].max():g}")

    # --- measure separation, before anything is simulated ------------------
    separation = check_measure_separation(
        snap.historical_drift, snap.risk_free_rate, snap.dividend_yield)
    print(f"\n  measure separation: {separation}")
    if not separation.passed:
        print("  Refusing to continue: risk would be simulated under the "
              "pricing measure.")
        return 1

    # --- candidates --------------------------------------------------------
    candidates, rejects = select_candidate_puts(
        chain, snap.spot, expiry=snap.expiry,
        days_to_expiry=snap.days_to_expiry, as_of=snap.as_of)

    print(f"\n  {len(candidates)} candidate contracts "
          f"({len(rejects)} rejected for quote quality)")
    for reject in rejects:
        print(f"    rejected K={reject['strike']:g}: {reject['reason']}")

    header = (f"\n  {'target':>7} {'strike':>8} {'moneyness':>10} {'bid':>7} "
              f"{'ask':>7} {'mid':>8} {'spread%':>8} {'source':>7} "
              f"{'open int':>9}")
    print(header)
    print("  " + "-" * (len(header) - 3))
    for c in candidates:
        spread = (c.ask - c.bid) / c.mid * 100.0 if c.mid else float("nan")
        print(f"  {c.target_moneyness:>7.3f} {c.strike:>8.1f} "
              f"{c.moneyness:>10.4f} {c.bid:>7.2f} {c.ask:>7.2f} "
              f"{c.mid:>8.3f} {spread:>8.2f} {c.price_source:>7} "
              f"{c.open_interest:>9,.0f}")

    # --- one scenario set for every candidate ------------------------------
    # Same seed as the baseline portfolio experiment, so the unhedged benchmark
    # here is the very same distribution that study reported.
    horizon_spots = simulate_horizon_scenarios(
        S0=snap.spot, real_world_drift=snap.historical_drift,
        sigma=snap.historical_volatility,
        horizon_days=config.RISK_HORIZON_DAYS,
        n_scenarios=config.N_RISK_SCENARIOS,
        trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
        seed=config.SEED + 2, antithetic=True)

    print(f"\n  {config.N_RISK_SCENARIOS:,} real-world scenarios over "
          f"{config.RISK_HORIZON_DAYS} trading days, shared by every candidate")
    print(f"    pricing drift r - q = "
          f"{snap.risk_free_rate - snap.dividend_yield:+.6f}   "
          f"risk drift mu = {snap.historical_drift:+.6f}")

    grid_spots = moneyness_grid(snap.spot, config.GRID_MIN_MONEYNESS,
                                config.GRID_MAX_MONEYNESS, config.GRID_N_POINTS)

    def progress(index, candidate):
        print(f"\n  [{index + 1}/{len(candidates)}] K={candidate.strike:g} "
              f"({candidate.moneyness:.1%} of spot)", flush=True)

    started = time.perf_counter()
    rows, baseline, extras = evaluate_candidates(
        candidates, snap, horizon_spots, t_remaining, grid_spots,
        shares=config.SHARES, contracts=1,
        multiplier=config.CONTRACT_MULTIPLIER, levels=LEVELS,
        lsmc_paths=config.LSMC_N_PATHS, lsmc_steps=config.LSMC_N_STEPS,
        lsmc_degree=config.LSMC_DEGREE, binomial_steps=config.BINOMIAL_N_STEPS,
        grid_paths=config.GRID_N_PATHS, grid_steps=config.GRID_N_STEPS,
        grid_degree=config.GRID_DEGREE, seed=config.SEED,
        interpolation_method=config.INTERPOLATION_METHOD,
        cache_dir=GRID_CACHE, rebuild_grids=args.rebuild_grids,
        progress=progress)
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame(rows)
    print(f"\n  evaluated {len(frame)} candidates in {elapsed:.1f}s "
          f"({sum(r['grid_source'] == 'cached' for r in rows)} grids reused)")

    # --- calibration and pricing agreement ---------------------------------
    head = (f"\n  {'strike':>8} {'sigma':>9} {'chain IV':>9} {'LSMC':>9} "
            f"{'CRR':>9} {'rel err':>9} {'market mid':>11} {'ask':>8}")
    print(head)
    print("  " + "-" * (len(head) - 3))
    for _, r in frame.iterrows():
        print(f"  {r['strike']:>8.1f} {r['sigma']:>9.6f} {r['quoted_iv']:>9.4f} "
              f"{r['lsmc_price']:>9.4f} {r['binomial_price']:>9.4f} "
              f"{r['rel_error_vs_binomial']:>9.4%} {r['market_mid']:>11.3f} "
              f"{r['market_ask']:>8.2f}")
    print("\n   The calibrated sigma rises as the strike falls: that is the put")
    print("   skew, priced from real quotes rather than assumed. Our sigma sits")
    print("   above the chain's own IV field because ours reproduces an")
    print("   AMERICAN price, which is worth more than the European one that")
    print("   field is built on, so less volatility is needed there.")

    # --- interpolation audit, per candidate --------------------------------
    if not args.skip_interpolation_check:
        print(f"\n  interpolation audit "
              f"({config.INTERPOLATION_CHECK_POINTS} direct-priced check "
              f"points per candidate)")
        audit_rows = []
        rng = np.random.default_rng(config.SEED + 7)
        for extra in extras:
            grid = extra["grid"]
            spots = random_check_spots(grid, config.INTERPOLATION_CHECK_POINTS,
                                       rng)
            # The SAME seed the grid was built with. Re-drawing the paths
            # instead would add the difference between two independent
            # 200,000-path runs -- about 0.07 at this sample size -- to a
            # quantity that is supposed to isolate interpolation error alone.
            reference = price_at_spots_directly(spots, grid, seed=config.SEED)
            for result in assess_interpolation_accuracy(
                    grid, spots, reference,
                    methods=(config.INTERPOLATION_METHOD,)):
                result["strike"] = grid.strike
                audit_rows.append(result)
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(INTERP_CSV, index=False)
        print(f"    {'strike':>8} {'max abs':>10} {'mean abs':>10} {'bias':>10}")
        for _, a in audit.iterrows():
            print(f"    {a['strike']:>8.1f} {a['max_absolute_error']:>10.4f} "
                  f"{a['mean_absolute_error']:>10.4f} {a['bias']:>+10.4f}")
        worst_interp = float(audit["max_absolute_error"].max())
        worst_bias = float(audit["bias"].abs().max())
        worst_mean = float(audit["mean_absolute_error"].max())
        print(f"\n    worst single point {worst_interp:.4f} per share. That "
              f"max is not what the answer depends on:")
        print("    VaR is a quantile and CVaR a tail average over 50,000")
        print("    scenarios, so a signed bias moves them and a scattered")
        print("    pointwise error largely cancels. The bias and the mean are")
        print("    checked below; the max is reported so nothing is hidden.")
    else:
        audit = pd.DataFrame()
        worst_interp = worst_bias = worst_mean = float("nan")

    # --- ranking -----------------------------------------------------------
    frame, winners = rank_candidates(frame, args.protection_weight,
                                     args.cost_weight)
    frame.to_csv(CANDIDATES_CSV, index=False)

    base = baseline_risk_row(baseline, LEVELS)
    print(f"\n  unhedged benchmark: {config.SHARES} SPY shares, "
          f"${baseline.initial_value:,.2f}")
    print(f"    95% VaR ${base['var_95_dollars']:,.2f}   "
          f"95% CVaR ${base['cvar_95_dollars']:,.2f}   "
          f"99% VaR ${base['var_99_dollars']:,.2f}   "
          f"99% CVaR ${base['cvar_99_dollars']:,.2f}")

    # --- the table the decision rests on -----------------------------------
    head = (f"\n  {'strike':>8} {'money':>7} {'ask':>7} {'premium':>9} "
            f"{'cost %':>7} {'99% VaR':>9} {'99% CVaR':>9} {'CVaR/$':>8} "
            f"{'pareto':>7}")
    print(head)
    print("  " + "-" * (len(head) - 3))
    for _, r in frame.iterrows():
        print(f"  {r['strike']:>8.1f} {r['moneyness']:>7.3f} "
              f"{r['ask']:>7.2f} {r['premium_cost']:>9,.2f} "
              f"{r['hedge_cost_percent']:>7.2f} "
              f"{r['var_99_reduction']:>8.2f}% {r['cvar_99_reduction']:>8.2f}% "
              f"{r['cvar_99_saved_per_premium_dollar']:>8.2f} "
              f"{'yes' if r['pareto_efficient'] else 'no':>7}")

    print("\n   Reductions are on the DOLLAR basis: every candidate is measured")
    print("   against the same unhedged benchmark, so the comparison between")
    print("   strikes is like for like. The percent-of-own-value basis is in")
    print("   the CSV as *_reduction_pct_basis.")

    # --- recommendations ---------------------------------------------------
    print(f"\n  balanced score = {args.protection_weight:.2f} x protection "
          f"+ {args.cost_weight:.2f} x cost, each min-max scaled across these "
          f"{len(frame)} candidates")

    labels = {
        "cheapest": "Cheapest hedge (lowest premium)",
        "strongest": "Strongest protection (max 99% CVaR reduction)",
        "most_efficient": "Best efficiency (max 99% CVaR saved per $1)",
        "balanced": "Balanced hedge (weighted score)",
    }
    ranking_rows = []
    print()
    for key, row in winners.items():
        print(f"  {labels[key]:<48} K={row['strike']:g} "
              f"({row['moneyness']:.1%})")
        print(f"  {'':<48} premium ${row['premium_cost']:,.2f} "
              f"({row['hedge_cost_percent']:.2f}%), "
              f"99% CVaR -{row['cvar_99_reduction']:.2f}%, "
              f"${row['cvar_99_saved_per_premium_dollar']:.2f} saved per $1")
        ranking_rows.append({
            "category": key, "description": labels[key],
            "strike": row["strike"], "moneyness": row["moneyness"],
            "ask": row["ask"], "premium_cost": row["premium_cost"],
            "hedge_cost_percent": row["hedge_cost_percent"],
            "var_99_reduction": row["var_99_reduction"],
            "cvar_99_reduction": row["cvar_99_reduction"],
            "cvar_99_saved_per_premium_dollar":
                row["cvar_99_saved_per_premium_dollar"],
            "balanced_score": row["balanced_score"],
            "pareto_efficient": bool(row["pareto_efficient"]),
            "protection_weight": args.protection_weight,
            "cost_weight": args.cost_weight,
        })
    pd.DataFrame(ranking_rows).to_csv(RANKINGS_CSV, index=False)

    frontier_columns = [
        "strike", "moneyness", "bid", "ask", "mid", "premium_cost",
        "hedge_cost_percent", "sigma", "lsmc_price", "binomial_price",
        "var_95_reduction", "cvar_95_reduction", "var_99_reduction",
        "cvar_99_reduction", "cvar_95_saved_per_premium_dollar",
        "cvar_99_saved_per_premium_dollar", "protection_score", "cost_score",
        "balanced_score", "pareto_efficient",
    ]
    frame[frontier_columns].to_csv(FRONTIER_CSV, index=False)

    # --- sanity checks ------------------------------------------------------
    print("\n  sanity checks:")
    strikes = frame["strike"].to_numpy()
    listed = set(chain["strike"].astype(float))
    cheapest = winners["cheapest"]
    strongest = winners["strongest"]
    efficient = winners["most_efficient"]
    pareto = frame["pareto_efficient"].to_numpy()
    costs = frame["premium_cost"].to_numpy()
    protection = frame["cvar_99_reduction"].to_numpy()

    dominated_on_frontier = 0
    for i in np.flatnonzero(pareto):
        better = ((costs <= costs[i]) & (protection >= protection[i])
                  & ((costs < costs[i]) | (protection > protection[i])))
        dominated_on_frontier += int(np.any(better))

    checks = [
        Check("every candidate strike is really listed",
              all(float(k) in listed for k in strikes),
              f"{len(strikes)} strikes checked against the chain"),
        Check("candidate strikes are unique",
              len(set(strikes)) == len(strikes),
              f"{len(set(strikes))} distinct of {len(strikes)}"),
        Check("cheapest really has the minimum premium",
              np.isclose(cheapest["premium_cost"], costs.min()),
              f"${cheapest['premium_cost']:,.2f} vs min ${costs.min():,.2f}"),
        Check("strongest really has the maximum 99% CVaR reduction",
              np.isclose(strongest["cvar_99_reduction"], protection.max()),
              f"{strongest['cvar_99_reduction']:.4f}% vs max "
              f"{protection.max():.4f}%"),
        Check("efficiency winner really maximises CVaR saved per dollar",
              np.isclose(efficient["cvar_99_saved_per_premium_dollar"],
                         frame["cvar_99_saved_per_premium_dollar"].max()),
              f"${efficient['cvar_99_saved_per_premium_dollar']:.4f} per $1"),
        Check("no point on the Pareto frontier is dominated",
              dominated_on_frontier == 0,
              f"{int(pareto.sum())} efficient, {dominated_on_frontier} dominated"),
        Check("premium is ask x contract multiplier for every candidate",
              np.allclose(costs, frame["ask"].to_numpy()
                          * config.CONTRACT_MULTIPLIER),
              "acquisition cost uses the ask, not the mid"),
    ]
    for _, r in frame.iterrows():
        for level in LEVELS:
            key = f"{level:.0%}".replace("%", "")
            checks.append(check_risk_measures(r[f"var_{key}_dollars"],
                                              r[f"cvar_{key}_dollars"], level))
    if not audit.empty:
        # What the risk numbers actually inherit from the interpolant is its
        # bias, plus whatever the mean error contributes once 50,000 scenarios
        # have averaged. Both are checked against the size of the decision they
        # could distort, rather than against a round number.
        smallest_cvar = float(frame["cvar_99_dollars"].min())
        mean_dollars = worst_mean * config.CONTRACT_MULTIPLIER
        checks.append(Check(
            "interpolation bias stays under half a cent per share",
            worst_bias < 0.005,
            f"worst |bias| {worst_bias:.5f} across {len(audit)} grids"))
        checks.append(Check(
            "interpolation cannot move any 99% CVaR by 0.1%",
            mean_dollars < 0.001 * smallest_cvar,
            f"${mean_dollars:.2f} per contract against the smallest "
            f"99% CVaR ${smallest_cvar:,.2f}"))

    all_passed = report(checks)

    # --- figures ------------------------------------------------------------
    plots.apply_style()
    highlight = int(np.flatnonzero(
        frame["strike"].to_numpy() == winners["balanced"]["strike"])[0])

    fig, _ = plots_hedge.plot_protection_cost_frontier(
        frame["hedge_cost_percent"], frame["cvar_99_reduction"],
        frame["strike"], frame["moneyness"], pareto, highlight=highlight,
        highlight_label="balanced choice",
        title="Which put protects best for its cost? (99% CVaR, 10-day horizon)")
    plots.save(fig, config.FIGURES_DIR, "13_protection_cost_frontier")

    fig, _ = plots_hedge.plot_efficiency_by_strike(
        frame["strike"], frame["cvar_95_saved_per_premium_dollar"],
        frame["cvar_99_saved_per_premium_dollar"])
    plots.save(fig, config.FIGURES_DIR, "14_cvar_saved_per_dollar")

    fig, _ = plots_hedge.plot_reduction_by_moneyness(
        frame["moneyness"], {
            # Grouped by confidence level so the legend reads down the same
            # ordering the colour ramp encodes.
            "95% VaR": frame["var_95_reduction"].to_numpy(),
            "95% CVaR": frame["cvar_95_reduction"].to_numpy(),
            "99% VaR": frame["var_99_reduction"].to_numpy(),
            "99% CVaR": frame["cvar_99_reduction"].to_numpy(),
        })
    plots.save(fig, config.FIGURES_DIR, "15_reduction_by_moneyness")

    print(f"\n  tables  {CANDIDATES_CSV.name}, {RANKINGS_CSV.name}, "
          f"{FRONTIER_CSV.name}, {INTERP_CSV.name}")
    print("  figures 13_protection_cost_frontier, 14_cvar_saved_per_dollar, "
          "15_reduction_by_moneyness")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
