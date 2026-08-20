#!/usr/bin/env python
"""PHASES 8 and 9: portfolio simulation, then VaR and CVaR.

Scope sections 12, 13, 14 and 16.

    Portfolio A   100 SPY shares
    Portfolio B   100 SPY shares + 1 American SPY put

50,000 real-world scenarios for the SPY price at the horizon, the hedge valued
off the interpolated LSMC pricing grid, and the resulting loss distribution for
each portfolio.

PHASE 9 then measures 95% and 99% VaR and CVaR on those losses and reports
how much of each the hedge removes -- computed, never assumed.

The measure separation the scope warns about is checked before anything is
simulated: pricing used the risk-neutral drift r - q, this simulation uses the
historical mu, and the run stops if those two ever coincide.

    python experiments/portfolio_risk.py
    python experiments/portfolio_risk.py --rebuild-grid
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.binomial import crr_american_put
from src.interpolation import PricingGridInterpolator
from src.lsmc import price_american_put_lsmc
from src.market_data import MarketSnapshot
from src.portfolio import (compare, hedge_coverage, protective_put_portfolio,
                           unhedged_portfolio)
from src.pricing_grid import (build_pricing_grid, grid_matches, load_grid,
                              moneyness_grid, save_grid)
from src.risk_simulation import (horizon_in_years, scenario_summary,
                                 simulate_horizon_scenarios)
from src.sanity import (Check, check_measure_separation, check_risk_measures,
                        report)
from src.var_cvar import (bootstrap_risk_measures, cvar_by_minimisation,
                          risk_reduction, risk_table)

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
SUMMARY_CSV = config.TABLES_DIR / "portfolio_loss_summary.csv"
RISK_CSV = config.TABLES_DIR / "risk_var_cvar.csv"
RISK_PCT_CSV = config.TABLES_DIR / "risk_var_cvar_percent.csv"
SCENARIOS_CSV = config.TABLES_DIR / "portfolio_scenarios_sample.csv"
LOSSES_NPZ = config.DATA_DIR / "portfolio_losses.npz"


def get_grid(snap, T_remaining, rebuild):
    """Load the cached pricing grid, or build it if it is missing or stale."""
    if not rebuild and GRID_JSON.exists():
        grid = load_grid(GRID_JSON)
        if grid_matches(grid, snap.strike, T_remaining, snap.risk_free_rate,
                        snap.historical_volatility, snap.dividend_yield):
            return grid, "cached"
        print("  cached grid was built for different inputs; rebuilding")

    spots = moneyness_grid(snap.spot, config.GRID_MIN_MONEYNESS,
                           config.GRID_MAX_MONEYNESS, config.GRID_N_POINTS)
    grid = build_pricing_grid(
        spots=spots, K=snap.strike, T_remaining=T_remaining,
        r=snap.risk_free_rate, sigma=snap.historical_volatility,
        q=snap.dividend_yield, n_paths=config.GRID_N_PATHS,
        n_steps=config.GRID_N_STEPS, degree=config.GRID_DEGREE,
        seed=config.SEED, antithetic=True)
    save_grid(grid, GRID_JSON)
    return grid, "rebuilt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-grid", action="store_true",
                        help="rebuild the pricing grid instead of using the cache")
    args = parser.parse_args()

    if not SNAPSHOT_JSON.exists():
        print(f"No market snapshot at {SNAPSHOT_JSON}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    horizon_years = horizon_in_years(config.RISK_HORIZON_DAYS,
                                     config.TRADING_DAYS_PER_YEAR)
    T_remaining = snap.time_to_expiry - horizon_years
    if T_remaining <= 0:
        print("The risk horizon reaches past expiry; pick a longer contract.")
        return 1

    print("=" * 78)
    print("PHASE 8  Portfolio simulation over the 10-day risk horizon")
    print("=" * 78)

    # --- measure separation, before anything is simulated ------------------
    separation = check_measure_separation(
        snap.historical_drift, snap.risk_free_rate, snap.dividend_yield)
    print("\n  measure separation (scope section 7):")
    print(f"   {separation}")
    if not separation.passed:
        print("\n  Refusing to continue: the risk simulation would be running")
        print("  under the pricing measure, which is the mistake the scope")
        print("  singles out as most likely and most damaging.")
        return 1
    print(f"    pricing  drift r - q = "
          f"{snap.risk_free_rate - snap.dividend_yield:+.6f}  (risk-neutral)")
    print(f"    risk     drift mu    = {snap.historical_drift:+.6f}  "
          f"(real world, from {snap.n_history_days} days of SPY history)")

    # --- today's hedge value ----------------------------------------------
    contract = dict(S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
                    r=snap.risk_free_rate, sigma=snap.historical_volatility,
                    q=snap.dividend_yield)
    today = price_american_put_lsmc(
        n_paths=config.GRID_N_PATHS, n_steps=config.GRID_N_STEPS,
        degree=config.GRID_DEGREE, seed=config.SEED, antithetic=True,
        **contract)
    today_binomial = crr_american_put(n_steps=config.BINOMIAL_N_STEPS,
                                      **contract)

    print(f"\n  put value today (T = {snap.time_to_expiry:.4f} yr):")
    print(f"    LSMC      {today.price:8.4f}  (s.e. {today.std_error:.4f})")
    print(f"    binomial  {today_binomial:8.4f}")
    print(f"    market    {snap.market_put_price:8.4f}  ({snap.price_source})")

    # --- pricing grid at the horizon --------------------------------------
    grid, grid_source = get_grid(snap, T_remaining, args.rebuild_grid)
    interpolate = PricingGridInterpolator(grid, config.INTERPOLATION_METHOD)
    print(f"\n  pricing grid ({grid_source}): {grid.spots.size} nodes, "
          f"{grid.spot_range[0]:.2f} .. {grid.spot_range[1]:.2f}, "
          f"T = {grid.time_to_expiry:.4f} yr at the horizon")

    # --- real-world scenarios ---------------------------------------------
    spots_h = simulate_horizon_scenarios(
        S0=snap.spot, real_world_drift=snap.historical_drift,
        sigma=snap.historical_volatility,
        horizon_days=config.RISK_HORIZON_DAYS,
        n_scenarios=config.N_RISK_SCENARIOS,
        trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
        seed=config.SEED + 2, antithetic=True)

    scenarios = scenario_summary(spots_h, snap.spot)
    print(f"\n  {config.N_RISK_SCENARIOS:,} scenarios, "
          f"{config.RISK_HORIZON_DAYS} trading days "
          f"({horizon_years:.4f} yr):")
    print(f"    spot   mean {scenarios['mean_spot']:8.2f}   "
          f"1st pct {scenarios['p01_spot']:8.2f}   "
          f"99th pct {scenarios['p99_spot']:8.2f}   "
          f"range [{scenarios['min_spot']:.2f}, {scenarios['max_spot']:.2f}]")
    print(f"    return mean {scenarios['mean_return']:+8.4%}   "
          f"std {scenarios['std_return']:8.4%}")

    coverage = interpolate.out_of_range(spots_h)
    if coverage["total"]:
        print(f"    NOTE: {coverage['total']} scenarios "
              f"({coverage['fraction']:.4%}) fell outside the grid "
              f"[{coverage['grid_min']:.2f}, {coverage['grid_max']:.2f}] "
              "and were valued by the no-arbitrage bounds")
    else:
        print(f"    all scenarios landed inside the grid")

    put_h = interpolate(spots_h)

    # --- the two portfolios ------------------------------------------------
    portfolio_a = unhedged_portfolio(config.SHARES, snap.spot, spots_h)
    portfolio_b = protective_put_portfolio(
        config.SHARES, snap.spot, spots_h, today.price, put_h,
        contracts=1, multiplier=config.CONTRACT_MULTIPLIER)

    print(f"\n  hedge coverage: "
          f"{hedge_coverage(config.SHARES, 1, config.CONTRACT_MULTIPLIER):.0%} "
          f"of the {config.SHARES}-share position")
    print(f"\n  {'':<12} {'V_0':>12} {'stock':>12} {'put':>10} "
          f"{'mean V_h':>12} {'mean loss':>11} {'std loss':>10}")
    print("  " + "-" * 72)
    for p in (portfolio_a, portfolio_b):
        s = p.summary()
        print(f"  {p.name:<12} {s['initial_value']:>12,.2f} "
              f"{p.components['stock_now']:>12,.2f} "
              f"{p.components['put_now']:>10,.2f} "
              f"{s['mean_horizon_value']:>12,.2f} "
              f"{s['mean_loss']:>11,.2f} {s['std_loss']:>10,.2f}")

    print(f"\n  loss distribution ({config.N_RISK_SCENARIOS:,} scenarios, "
          "positive = money lost)")
    print(f"  {'':<12} {'P(loss)':>9} {'mean %':>9} {'std %':>9} "
          f"{'worst $':>12} {'worst %':>9} {'best $':>12}")
    print("  " + "-" * 76)
    for p in (portfolio_a, portfolio_b):
        s = p.summary()
        print(f"  {p.name:<12} {s['probability_of_loss']:>9.2%} "
              f"{s['mean_percent_loss']:>9.4%} {s['std_percent_loss']:>9.4%} "
              f"{s['max_loss']:>12,.2f} {s['worst_percent_loss']:>9.2%} "
              f"{s['min_loss']:>12,.2f}")

    summary = compare(portfolio_a, portfolio_b)
    summary.to_csv(SUMMARY_CSV, index=False)

    order = np.argsort(spots_h)[::max(1, config.N_RISK_SCENARIOS // 500)]
    pd.DataFrame({
        "spot_horizon": spots_h[order],
        "put_price_horizon": put_h[order],
        "value_unhedged": portfolio_a.horizon_values[order],
        "value_protected": portfolio_b.horizon_values[order],
        "loss_unhedged": portfolio_a.losses[order],
        "loss_protected": portfolio_b.losses[order],
    }).to_csv(SCENARIOS_CSV, index=False)

    np.savez_compressed(
        LOSSES_NPZ, spots_horizon=spots_h, put_prices_horizon=put_h,
        losses_unhedged=portfolio_a.losses,
        losses_protected=portfolio_b.losses,
        percent_losses_unhedged=portfolio_a.percent_losses,
        percent_losses_protected=portfolio_b.percent_losses,
        initial_unhedged=portfolio_a.initial_value,
        initial_protected=portfolio_b.initial_value)

    # --- sanity checks -----------------------------------------------------
    breakeven = np.argmin(np.abs(spots_h - snap.spot))
    print("\n  sanity checks:")
    checks = [
        separation,
        Check("hedge caps the worst percentage loss",
              portfolio_b.summary()["worst_percent_loss"]
              < portfolio_a.summary()["worst_percent_loss"],
              f"{portfolio_b.summary()['worst_percent_loss']:.2%} vs "
              f"{portfolio_a.summary()['worst_percent_loss']:.2%}"),
        Check("protected portfolio starts out worth more (it owns the put)",
              portfolio_b.initial_value > portfolio_a.initial_value,
              f"{portfolio_b.initial_value:,.2f} vs "
              f"{portfolio_a.initial_value:,.2f}"),
        Check("put value rises as the spot falls",
              bool(np.all(np.diff(put_h[np.argsort(spots_h)]) <= 1e-9)),
              "monotone across all scenarios"),
        Check("put never worth less than intrinsic at the horizon",
              bool(np.all(put_h >= np.maximum(snap.strike - spots_h, 0.0) - 1e-9)),
              f"min slack "
              f"{float(np.min(put_h - np.maximum(snap.strike - spots_h, 0.0))):.6f}"),
        Check("unhedged loss is exactly linear in the spot move",
              bool(np.allclose(portfolio_a.losses,
                               config.SHARES * (snap.spot - spots_h))),
              "L_A = 100 (S_0 - S_h)"),
        Check("scenario mean return matches the drift over the horizon",
              abs(scenarios["mean_return"]
                  - (np.exp(snap.historical_drift * horizon_years) - 1.0)) < 5e-4,
              f"{scenarios['mean_return']:+.4%} vs theory "
              f"{np.exp(snap.historical_drift * horizon_years) - 1.0:+.4%}"),
    ]
    all_passed = report(checks)

    a, b = portfolio_a.summary(), portfolio_b.summary()
    print("\n  Reading the distributions:")
    print(f"   * The put costs {portfolio_b.components['put_now']:,.2f} up front, "
          f"{portfolio_b.components['put_now'] / portfolio_a.initial_value:.2%} "
          "of the share position.")
    print(f"   * In the scenario closest to no move (S_h = "
          f"{spots_h[breakeven]:.2f}) the protected portfolio loses "
          f"{portfolio_b.losses[breakeven]:,.2f}")
    print(f"     against {portfolio_a.losses[breakeven]:,.2f} unhedged: that gap "
          "is the option's time decay over ten days.")
    print(f"   * Worst case the hedge cuts the loss from "
          f"{a['worst_percent_loss']:.2%} to {b['worst_percent_loss']:.2%}, "
          "but the")
    print(f"     average loss is {b['mean_percent_loss']:.4%} hedged against "
          f"{a['mean_percent_loss']:.4%} unhedged -- protection is not free.")
    # ==================================================================
    # PHASE 9  VaR and CVaR
    # ==================================================================
    print("\n" + "=" * 78)
    print("PHASE 9  Value-at-Risk and Conditional Value-at-Risk")
    print("=" * 78)

    levels = tuple(config.CONFIDENCE_LEVELS)
    dollars = risk_table([portfolio_a, portfolio_b], levels)
    percent = risk_table([portfolio_a, portfolio_b], levels,
                         use_percentage=True)
    dollars.to_csv(RISK_CSV, index=False)
    percent.to_csv(RISK_PCT_CSV, index=False)

    boot_rng = np.random.default_rng(config.SEED + 3)
    boot_a = bootstrap_risk_measures(portfolio_a.losses, levels, 500, boot_rng)
    boot_b = bootstrap_risk_measures(portfolio_b.losses, levels, 500, boot_rng)

    def key(level, stat):
        return f"{stat}_{level:.0%}".replace("%", "")

    header = (f"  {'portfolio':<12} {'95% VaR':>12} {'95% CVaR':>12} "
              f"{'99% VaR':>12} {'99% CVaR':>12}")

    print(f"\n  losses in dollars  ({config.N_RISK_SCENARIOS:,} scenarios, "
          f"{config.RISK_HORIZON_DAYS}-day horizon)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate([portfolio_a.name, portfolio_b.name]):
        row = dollars.iloc[i]
        print(f"  {name:<12} {row[key(0.95, 'var')]:>12,.2f} "
              f"{row[key(0.95, 'cvar')]:>12,.2f} "
              f"{row[key(0.99, 'var')]:>12,.2f} "
              f"{row[key(0.99, 'cvar')]:>12,.2f}")
    red = dollars.iloc[-1]
    print(f"  {'reduction':<12} {red[key(0.95, 'var')]:>11.2f}% "
          f"{red[key(0.95, 'cvar')]:>11.2f}% {red[key(0.99, 'var')]:>11.2f}% "
          f"{red[key(0.99, 'cvar')]:>11.2f}%")

    print("\n  bootstrap 95% intervals for CVaR (500 resamples)")
    for name, boot in ((portfolio_a.name, boot_a), (portfolio_b.name, boot_b)):
        parts = []
        for level in levels:
            k = key(level, "cvar")
            parts.append(f"{level:.0%} [{boot[k + '_ci_low']:,.0f}, "
                         f"{boot[k + '_ci_high']:,.0f}]")
        print(f"    {name:<12} " + "   ".join(parts))

    print("\n  losses as a percentage of each portfolio's own initial value")
    print("  (the fair comparison: the protected portfolio starts out worth")
    print(f"   {portfolio_b.initial_value - portfolio_a.initial_value:,.2f} "
          "more, because it owns the put)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate([portfolio_a.name, portfolio_b.name]):
        row = percent.iloc[i]
        print(f"  {name:<12} {row[key(0.95, 'var')]:>11.4%} "
              f"{row[key(0.95, 'cvar')]:>11.4%} {row[key(0.99, 'var')]:>11.4%} "
              f"{row[key(0.99, 'cvar')]:>11.4%}")
    redp = percent.iloc[-1]
    print(f"  {'reduction':<12} {redp[key(0.95, 'var')]:>11.2f}% "
          f"{redp[key(0.95, 'cvar')]:>11.2f}% {redp[key(0.99, 'var')]:>11.2f}% "
          f"{redp[key(0.99, 'cvar')]:>11.2f}%")

    # --- Rockafellar-Uryasev cross-check ----------------------------------
    print("\n  Rockafellar & Uryasev Theorem 1 cross-check")
    print("  CVaR found by minimising F_beta(alpha) must equal the empirical")
    print("  tail mean, and the minimiser must come back as the VaR.")
    ru_checks = []
    for p in (portfolio_a, portfolio_b):
        for level in levels:
            ru = cvar_by_minimisation(p.losses, level)
            empirical = risk_table([p], (level,)).iloc[0][key(level, "cvar")]
            print(f"    {p.name:<12} {level:.0%}:  min F = {ru['cvar']:>10,.2f}"
                  f"   empirical CVaR = {empirical:>10,.2f}"
                  f"   alpha* = {ru['alpha_star']:>10,.2f}"
                  f"   VaR = {ru['empirical_var']:>10,.2f}")
            ru_checks.append(Check(
                f"{p.name} {level:.0%}: minimised F equals empirical CVaR",
                abs(ru["cvar"] - empirical) < 0.01 * max(abs(empirical), 1.0),
                f"{ru['cvar']:,.2f} vs {empirical:,.2f}"))
            ru_checks.append(Check(
                f"{p.name} {level:.0%}: minimiser recovers the VaR",
                abs(ru["alpha_star"] - ru["empirical_var"])
                < 0.01 * max(abs(ru["empirical_var"]), 1.0),
                f"alpha* {ru['alpha_star']:,.2f} vs VaR "
                f"{ru['empirical_var']:,.2f}"))

    print("\n  sanity checks (scope section 10):")
    risk_checks = []
    for i, p in enumerate((portfolio_a, portfolio_b)):
        for level in levels:
            row = dollars.iloc[i]
            risk_checks.append(check_risk_measures(
                row[key(level, "var")], row[key(level, "cvar")], level))
    risk_checks.extend(ru_checks)
    all_passed &= report(risk_checks)

    # --- the answer --------------------------------------------------------
    print("\n  Does the protective put reduce tail risk?")
    verdicts = []
    for level in levels:
        for stat, label in (("var", "VaR"), ("cvar", "CVaR")):
            pct = redp[key(level, stat)]
            verdicts.append(pct > 0)
            direction = "reduces" if pct > 0 else "INCREASES"
            print(f"    {level:.0%} {label:<5}: the hedge {direction} it by "
                  f"{abs(pct):.2f}%  (percentage-of-portfolio terms)")

    if all(verdicts):
        print("\n    Every measure improves. Reported as computed, not assumed.")
    else:
        print("\n    Not every measure improves. Reported as computed.")

    cost = portfolio_b.components["put_now"]
    saved_99 = (dollars.iloc[0][key(0.99, "cvar")]
                - dollars.iloc[1][key(0.99, "cvar")])
    worse_off = float((portfolio_b.losses > portfolio_a.losses).mean())
    print(f"\n    The hedge cost {cost:,.2f} up front and removes "
          f"{saved_99:,.2f} of 99% CVaR:")
    print(f"    {saved_99 / cost:.2f} dollars of extreme tail loss avoided per "
          "dollar spent. That is")
    print(f"    not a free lunch -- in {worse_off:.1%} of scenarios the hedged "
          "portfolio ends up")
    print("    behind the unhedged one, which is the premium being paid.")

    print(f"\n  tables saved to {SUMMARY_CSV.name}, {SCENARIOS_CSV.name}, "
          f"{RISK_CSV.name}, {RISK_PCT_CSV.name}")
    print(f"  loss distributions saved to {LOSSES_NPZ.name}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
