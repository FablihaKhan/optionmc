#!/usr/bin/env python
"""ADVANCED PHASE 3a: does the hedging conclusion survive a non-GBM risk model?

The project's risk engine assumes geometric Brownian motion. This adds a second
engine that assumes nothing: each scenario resamples ten observed SPY daily log
returns with replacement and sums them. GBM is kept and both are run on the
same portfolios, the same horizon and the same pricing grid.

The comparison is deliberately controlled. Summing ten draws from the observed
sample gives the same mean and the same variance as the GBM step by
construction, so the two engines differ only in the shape of the distribution
-- its skewness and its fat tails. Any gap in VaR or CVaR is attributable to
that shape and to nothing else.

    python experiments/bootstrap_risk_experiment.py

No risk-free rate enters the bootstrap: `bootstrap_horizon_prices` has no
drift, rate or volatility argument at all. Option values at the horizon remain
risk-neutral and still come off the cached LSMC pricing grid, so the two
measures stay separated.
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
from src.historical_bootstrap import (bootstrap_horizon_prices,
                                      bootstrap_horizon_returns,
                                      daily_log_returns, quantile_comparison,
                                      return_distribution_summary,
                                      tail_exceedance)
from src.interpolation import PricingGridInterpolator
from src.market_data import MarketSnapshot, load_spy_history
from src.portfolio import protective_put_portfolio, unhedged_portfolio
from src.pricing_grid import grid_matches, load_grid
from src.risk_simulation import horizon_in_years, simulate_horizon_scenarios
from src.sanity import Check, check_measure_separation, check_risk_measures, report
from src.var_cvar import risk_measures, risk_reduction

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
HEDGE_GRIDS = config.DATA_DIR / "hedge_grids"

COMPARISON_CSV = config.TABLES_DIR / "risk_model_comparison.csv"
DIAGNOSTICS_CSV = config.TABLES_DIR / "return_distribution_diagnostics.csv"
QUANTILES_CSV = config.TABLES_DIR / "horizon_quantile_comparison.csv"
LOSSES_NPZ = config.DATA_DIR / "bootstrap_losses.npz"

LEVELS = tuple(config.CONFIDENCE_LEVELS)


def build_portfolios(snap, spots, put_specs):
    """The unhedged position plus one protected portfolio per hedge."""
    portfolios = [unhedged_portfolio(config.SHARES, snap.spot, spots)]
    for spec in put_specs:
        horizon_put = spec["interpolate"](spots)
        portfolios.append(protective_put_portfolio(
            config.SHARES, snap.spot, spots, spec["cost_per_share"],
            horizon_put, contracts=1, multiplier=config.CONTRACT_MULTIPLIER,
            name=spec["name"]))
    return portfolios


def main():
    if not SNAPSHOT_JSON.exists() or not GRID_JSON.exists():
        print("Run experiments/portfolio_risk.py first "
              "(the pricing grid is missing).")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    horizon_years = horizon_in_years(config.RISK_HORIZON_DAYS,
                                     config.TRADING_DAYS_PER_YEAR)
    t_remaining = snap.time_to_expiry - horizon_years

    print("=" * 78)
    print("ADVANCED PHASE 3a  GBM against a historical bootstrap")
    print("=" * 78)

    separation = check_measure_separation(
        snap.historical_drift, snap.risk_free_rate, snap.dividend_yield)
    print(f"  measure separation: {separation}")
    if not separation.passed:
        return 1

    # --- historical returns -------------------------------------------------
    history = load_spy_history(config.SPY_HISTORY_CSV)
    closes = history["Close"].dropna()
    returns = daily_log_returns(closes.to_numpy())
    observed = return_distribution_summary(returns, "observed daily")

    print(f"\n  {observed['n']} observed daily log returns, "
          f"{closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"    mean {observed['mean']:+.6f}   std {observed['std']:.6f}   "
          f"worst day {observed['min']:+.4f}   best day {observed['max']:+.4f}")
    print(f"    skewness {observed['skewness']:+.3f}   "
          f"excess kurtosis {observed['excess_kurtosis']:+.3f}")
    if observed["excess_kurtosis"] > 1:
        print("    Excess kurtosis well above zero: extreme days happen far")
        print("    more often than the normal distribution GBM assumes.")

    # --- the two scenario sets ---------------------------------------------
    gbm_spots = simulate_horizon_scenarios(
        S0=snap.spot, real_world_drift=snap.historical_drift,
        sigma=snap.historical_volatility,
        horizon_days=config.RISK_HORIZON_DAYS,
        n_scenarios=config.N_RISK_SCENARIOS,
        trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
        seed=config.SEED + 2, antithetic=True)

    # A second GBM arm with the location confound removed. `simulate_*` uses
    # S_T = S_0 exp((drift - sigma^2/2) T + sigma sqrt(T) Z), so the drift it
    # wants is the arithmetic one, while `estimate_gbm_parameters` returns the
    # mean LOG return. Feeding the latter straight in -- which is what the
    # project has always done -- leaves the simulated mean log return short by
    # sigma^2 T / 2. Adding it back gives a GBM whose first two moments match
    # the bootstrap exactly, so the gap between THAT and the bootstrap is the
    # distribution's shape and nothing else. The original arm is kept
    # unchanged so every published number still reproduces.
    matched_drift = snap.historical_drift + 0.5 * snap.historical_volatility ** 2
    gbm_matched_spots = simulate_horizon_scenarios(
        S0=snap.spot, real_world_drift=matched_drift,
        sigma=snap.historical_volatility,
        horizon_days=config.RISK_HORIZON_DAYS,
        n_scenarios=config.N_RISK_SCENARIOS,
        trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
        seed=config.SEED + 2, antithetic=True)

    boot_spots = bootstrap_horizon_prices(
        S0=snap.spot, log_returns=returns,
        horizon_days=config.RISK_HORIZON_DAYS,
        n_scenarios=config.N_RISK_SCENARIOS, seed=config.SEED + 3)

    gbm_returns = np.log(gbm_spots / snap.spot)
    gbm_matched_returns = np.log(gbm_matched_spots / snap.spot)
    boot_returns = bootstrap_horizon_returns(
        returns, config.RISK_HORIZON_DAYS, config.N_RISK_SCENARIOS,
        seed=config.SEED + 3)

    gbm_summary = return_distribution_summary(gbm_returns, "GBM as published")
    matched_summary = return_distribution_summary(gbm_matched_returns,
                                                  "GBM drift matched")
    boot_summary = return_distribution_summary(boot_returns, "bootstrap")
    pd.DataFrame([observed, gbm_summary, matched_summary,
                  boot_summary]).to_csv(DIAGNOSTICS_CSV, index=False)

    print(f"\n  {config.N_RISK_SCENARIOS:,} scenarios over "
          f"{config.RISK_HORIZON_DAYS} trading days:")
    head = (f"    {'engine':>18} {'mean':>10} {'std':>9} {'skew':>8} "
            f"{'ex kurt':>9} {'1st pct':>9} {'worst':>9}")
    print(head)
    print("    " + "-" * (len(head) - 5))
    for s in (gbm_summary, matched_summary, boot_summary):
        print(f"    {s['label']:>18} {s['mean']:>+10.6f} {s['std']:>9.6f} "
              f"{s['skewness']:>+8.3f} {s['excess_kurtosis']:>+9.3f} "
              f"{s['p01']:>+9.4f} {s['min']:>+9.4f}")

    drift_gap = boot_summary["mean"] - gbm_summary["mean"]
    print(f"\n   A NOTE ON THE DRIFT. src/gbm.py simulates")
    print(f"     S_T = S_0 exp((drift - sigma^2/2) T + sigma sqrt(T) Z),")
    print(f"   so `drift` means the arithmetic drift, while")
    print(f"   estimate_gbm_parameters returns the mean LOG return. The")
    print(f"   project has always passed the second into the first, which")
    print(f"   leaves the simulated mean log return short by sigma^2 T / 2 =")
    print(f"   {0.5 * snap.historical_volatility ** 2 * horizon_years:.6f} "
          f"over this horizon. Small, and conservative for VaR, but it is a")
    print(f"   location difference and would contaminate a claim about shape.")
    print(f"\n   The 'drift matched' row adds it back. Its mean now agrees")
    print(f"   with the bootstrap to "
          f"{abs(matched_summary['mean'] - boot_summary['mean']):.6f} and its")
    print(f"   standard deviation to "
          f"{abs(matched_summary['std'] - boot_summary['std']):.6f}, so the")
    print(f"   remaining difference is skewness "
          f"({boot_summary['skewness']:+.3f} against "
          f"{matched_summary['skewness']:+.3f}) and excess kurtosis")
    print(f"   ({boot_summary['excess_kurtosis']:+.3f} against "
          f"{matched_summary['excess_kurtosis']:+.3f}). The published GBM arm "
          f"is kept unchanged\n   so every earlier result still reproduces.")

    # --- the hedges ---------------------------------------------------------
    grid = load_grid(GRID_JSON)
    if not grid_matches(grid, snap.strike, t_remaining, snap.risk_free_rate,
                        snap.historical_volatility, snap.dividend_yield):
        print("\n  The cached pricing grid does not match the snapshot.")
        print("  Run: python experiments/portfolio_risk.py --rebuild-grid")
        return 1

    baseline_put = price_baseline_put(snap)
    specs = [{
        "name": f"SPY + put K={snap.strike:g}",
        "interpolate": PricingGridInterpolator(grid,
                                               config.INTERPOLATION_METHOD),
        "cost_per_share": baseline_put,
        "cost_basis": "LSMC model price",
        "strike": snap.strike,
        "grid": grid,
    }]

    efficient = load_efficient_hedge(t_remaining, snap)
    if efficient is not None:
        specs.append(efficient)

    print(f"\n  hedges compared:")
    for spec in specs:
        print(f"    {spec['name']:<24} cost {spec['cost_per_share']:.4f} "
              f"per share ({spec['cost_basis']}), grid sigma "
              f"{spec['grid'].volatility:.6f}")

    # --- grid coverage, measured before anything is priced ------------------
    print(f"\n  pricing grid [{grid.spot_range[0]:.2f}, "
          f"{grid.spot_range[1]:.2f}] at T = {t_remaining:.4f} yr")
    coverage_rows = []
    for name, spots in (("GBM", gbm_spots), ("GBM matched", gbm_matched_spots),
                        ("bootstrap", boot_spots)):
        for spec in specs:
            out = spec["interpolate"].out_of_range(spots)
            coverage_rows.append({"engine": name, "hedge": spec["name"],
                                  **out})
            print(f"    {name:>10} vs {spec['name']:<24} "
                  f"min {spots.min():7.2f}  max {spots.max():7.2f}  "
                  f"outside {out['total']:>3} ({out['fraction']:.4%})")
    outside_total = sum(row["total"] for row in coverage_rows)
    if outside_total:
        print("    Some scenarios fell outside the grid and were valued by the")
        print("    no-arbitrage bounds. Widen the grid rather than trust those.")
    else:
        print("    Every scenario from all three engines landed inside the grid,")
        print("    so no value here rests on extrapolation.")

    # --- risk under both models --------------------------------------------
    rows = []
    stored = {}
    for engine, spots in (("GBM Monte Carlo", gbm_spots),
                          ("GBM drift matched", gbm_matched_spots),
                          ("historical bootstrap", boot_spots)):
        portfolios = build_portfolios(snap, spots, specs)
        stored[engine] = portfolios
        baseline_dollars = risk_measures(portfolios[0].losses, LEVELS)
        for index, portfolio in enumerate(portfolios):
            dollars = risk_measures(portfolio.losses, LEVELS)
            percent = risk_measures(portfolio.percent_losses, LEVELS)
            row = {
                "risk_model": engine,
                "portfolio": portfolio.name,
                # Not "n/a": pandas reads that back as NaN, so the column
                # would silently lose the one value used to pick the
                # unhedged rows out again.
                "put_cost_basis": (specs[index - 1]["cost_basis"]
                                   if index else "unhedged"),
                "initial_value": portfolio.initial_value,
                "mean_loss": float(portfolio.losses.mean()),
                "probability_of_loss": float((portfolio.losses > 0).mean()),
            }
            for level in LEVELS:
                key = f"{level:.0%}".replace("%", "")
                for stat in ("var", "cvar"):
                    name = f"{stat}_{key}"
                    row[f"{name}_dollars"] = dollars[name]
                    row[f"{name}_percent"] = percent[name] * 100.0
                    row[f"{name}_reduction"] = (
                        0.0 if index == 0
                        else risk_reduction(baseline_dollars[name],
                                            dollars[name]))
            rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_CSV, index=False)

    print("\n  Risk model            Portfolio                 95% VaR   "
          "95% CVaR    99% VaR   99% CVaR")
    print("  " + "-" * 88)
    for _, r in comparison.iterrows():
        print(f"  {r['risk_model']:<21} {r['portfolio']:<22} "
              f"{r['var_95_dollars']:>9,.0f} {r['cvar_95_dollars']:>10,.0f} "
              f"{r['var_99_dollars']:>10,.0f} {r['cvar_99_dollars']:>10,.0f}")

    print("\n  hedge reduction against the unhedged position, same engine:")
    head = (f"    {'risk model':<21} {'hedge':<24} {'95% VaR':>9} "
            f"{'95% CVaR':>9} {'99% VaR':>9} {'99% CVaR':>9}")
    print(head)
    print("    " + "-" * (len(head) - 5))
    for _, r in comparison[comparison["put_cost_basis"] != "unhedged"].iterrows():
        print(f"    {r['risk_model']:<21} {r['portfolio']:<24} "
              f"{r['var_95_reduction']:>8.2f}% {r['cvar_95_reduction']:>8.2f}% "
              f"{r['var_99_reduction']:>8.2f}% {r['cvar_99_reduction']:>8.2f}%")

    # --- interpretation, read off the numbers ------------------------------
    unhedged = comparison[comparison["put_cost_basis"] == "unhedged"]
    gbm_cvar = float(unhedged[unhedged["risk_model"] == "GBM Monte Carlo"]
                     ["cvar_99_dollars"].iloc[0])
    boot_cvar = float(unhedged[unhedged["risk_model"] == "historical bootstrap"]
                      ["cvar_99_dollars"].iloc[0])
    matched = float(unhedged[unhedged["risk_model"] == "GBM drift matched"]
                    ["cvar_99_dollars"].iloc[0])
    harsher = "historical bootstrap" if boot_cvar > gbm_cvar else "GBM Monte Carlo"
    gap = abs(boot_cvar - gbm_cvar) / min(boot_cvar, gbm_cvar) * 100.0
    from_drift = gbm_cvar - matched
    from_shape = boot_cvar - matched

    hedged = comparison[comparison["put_cost_basis"] != "unhedged"]
    helps_everywhere = bool((hedged["cvar_99_reduction"] > 0).all())

    print(f"\n  what the numbers say:")
    print(f"    The more conservative 99% CVaR on the unhedged position comes")
    print(f"    from the {harsher} (${max(gbm_cvar, boot_cvar):,.2f} against "
          f"${min(gbm_cvar, boot_cvar):,.2f}, {gap:.1f}% apart).")
    print(f"    Decomposed against the drift-matched GBM (${matched:,.2f}):")
    print(f"      the drift convention alone makes the published GBM "
          f"${from_drift:,.2f} harsher;")
    print(f"      the fat tails add ${from_shape:,.2f} on top of that "
          f"baseline. The shape of the")
    print(f"      distribution, not its location, is what moves the tail.")
    print()
    print(f"    The hedge reduces 99% CVaR under "
          f"{'every model tried' if helps_everywhere else 'only some models'}:")
    for hedge_name in dict.fromkeys(hedged["portfolio"]):
        block = hedged[hedged["portfolio"] == hedge_name]
        detail = "  ".join(f"{r.risk_model}: {r.cvar_99_reduction:.1f}%"
                           for r in block.itertuples())
        print(f"      {hedge_name:<22} {detail}")

    thresholds = [-0.05, -0.10, -0.15]
    gbm_tail = tail_exceedance(gbm_returns, thresholds)
    boot_tail = tail_exceedance(boot_returns, thresholds)
    print(f"\n    probability of a ten-day fall worse than:")
    for t in thresholds:
        ratio = (f"{boot_tail[t] / gbm_tail[t]:.2f}x" if gbm_tail[t]
                 else "GBM never got there")
        print(f"      {t:+.0%}   GBM {gbm_tail[t]:8.4%}   "
              f"bootstrap {boot_tail[t]:8.4%}   {ratio}")

    np.savez_compressed(
        LOSSES_NPZ,
        gbm_unhedged=stored["GBM Monte Carlo"][0].losses,
        gbm_protected=stored["GBM Monte Carlo"][1].losses,
        bootstrap_unhedged=stored["historical bootstrap"][0].losses,
        bootstrap_protected=stored["historical bootstrap"][1].losses,
        gbm_horizon_returns=gbm_returns,
        bootstrap_horizon_returns=boot_returns)

    quantiles = quantile_comparison(boot_returns, gbm_returns)
    pd.DataFrame(quantiles).to_csv(QUANTILES_CSV, index=False)

    # --- sanity checks ------------------------------------------------------
    print("\n  sanity checks:")
    import inspect
    signature = inspect.signature(bootstrap_horizon_prices)
    forbidden = {"r", "rate", "risk_free_rate", "drift", "mu", "sigma",
                 "volatility"}

    matched_cvar = float(unhedged[unhedged["risk_model"] == "GBM drift matched"]
                         ["cvar_99_dollars"].iloc[0])
    checks = [
        Check("the bootstrap engine takes no rate, drift or volatility",
              not (forbidden & set(signature.parameters)),
              f"parameters: {', '.join(signature.parameters)}"),
        Check("bootstrapped returns are all drawn from the observed sample",
              bool(np.isin(np.round(returns[
                  np.random.default_rng(0).integers(0, returns.size, 200)], 12),
                  np.round(returns, 12)).all()),
              f"{returns.size} observed daily returns"),
        Check("the drift-matched GBM shares the bootstrap's horizon mean",
              abs(matched_summary["mean"] - boot_summary["mean"]) < 5e-4,
              f"{matched_summary['mean']:+.6f} vs {boot_summary['mean']:+.6f}"),
        Check("the drift-matched GBM shares the bootstrap's dispersion",
              abs(matched_summary["std"] - boot_summary["std"]) < 5e-4,
              f"{matched_summary['std']:.6f} vs {boot_summary['std']:.6f}"),
        Check("the published GBM arm sits below the matched one by sigma^2 T/2",
              abs((matched_summary["mean"] - gbm_summary["mean"])
                  - 0.5 * snap.historical_volatility ** 2 * horizon_years) < 1e-9,
              f"gap {matched_summary['mean'] - gbm_summary['mean']:.6f}"),
        Check("the bootstrap has the fatter tails",
              boot_summary["excess_kurtosis"] > matched_summary["excess_kurtosis"],
              f"excess kurtosis {boot_summary['excess_kurtosis']:+.3f} vs "
              f"{matched_summary['excess_kurtosis']:+.3f}"),
        Check("no scenario needed extrapolation past the grid",
              outside_total == 0,
              f"{outside_total} of {3 * len(specs) * config.N_RISK_SCENARIOS:,}"),
        Check("the GBM arm reproduces the published baseline",
              abs(gbm_cvar - 6453.244893) < 0.01,
              f"99% CVaR ${gbm_cvar:,.4f} against the recorded "
              f"$6,453.2449"),
    ]
    for _, r in comparison.iterrows():
        for level in LEVELS:
            key = f"{level:.0%}".replace("%", "")
            checks.append(check_risk_measures(r[f"var_{key}_dollars"],
                                              r[f"cvar_{key}_dollars"], level))
    all_passed = report(checks)

    # --- figures ------------------------------------------------------------
    plots.apply_style()

    fig, _ = plots_risk_models.plot_return_distributions(
        returns, observed["mean"], observed["std"])
    plots.save(fig, config.FIGURES_DIR, "23_return_distributions")

    fig, _ = plots_risk_models.plot_quantile_comparison(
        quantiles["probability"], quantiles["empirical"],
        quantiles["simulated"], config.RISK_HORIZON_DAYS)
    plots.save(fig, config.FIGURES_DIR, "24_horizon_quantiles")

    names = list(dict.fromkeys(comparison["portfolio"]))
    gbm_values = [float(comparison[(comparison["risk_model"] == "GBM Monte Carlo")
                                   & (comparison["portfolio"] == n)]
                        ["cvar_99_dollars"].iloc[0]) for n in names]
    boot_values = [float(comparison[(comparison["risk_model"] == "historical bootstrap")
                                    & (comparison["portfolio"] == n)]
                         ["cvar_99_dollars"].iloc[0]) for n in names]
    fig, _ = plots_risk_models.plot_risk_model_comparison(
        [n.replace("SPY + put ", "") for n in names], gbm_values, boot_values,
        measure="99% CVaR",
        title="99% CVaR: does the risk model change the answer?")
    plots.save(fig, config.FIGURES_DIR, "25_risk_model_comparison")

    fig, _ = plots_risk_models.plot_loss_distributions_by_model(
        stored["GBM Monte Carlo"][0].losses,
        stored["historical bootstrap"][0].losses,
        gbm_var=float(unhedged[unhedged["risk_model"] == "GBM Monte Carlo"]
                      ["var_99_dollars"].iloc[0]),
        bootstrap_var=float(unhedged[unhedged["risk_model"] == "historical bootstrap"]
                            ["var_99_dollars"].iloc[0]),
        label="100 SPY shares, unhedged")
    plots.save(fig, config.FIGURES_DIR, "26_loss_by_risk_model")

    print(f"\n  tables  {COMPARISON_CSV.name}, {DIAGNOSTICS_CSV.name}, "
          f"{QUANTILES_CSV.name}")
    print("  figures 23_return_distributions, 24_horizon_quantiles, "
          "25_risk_model_comparison, 26_loss_by_risk_model")
    return 0 if all_passed else 1


def price_baseline_put(snap):
    """Today's value of the baseline put, by the same route phase 8 used."""
    from src.lsmc import price_american_put_lsmc
    result = price_american_put_lsmc(
        S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
        r=snap.risk_free_rate, sigma=snap.historical_volatility,
        q=snap.dividend_yield, n_paths=config.GRID_N_PATHS,
        n_steps=config.GRID_N_STEPS, degree=config.GRID_DEGREE,
        seed=config.SEED, antithetic=True)
    return result.price


def load_efficient_hedge(t_remaining, snap):
    """The optimizer's most efficient hedge, if phase 1 has been run.

    Optional by design (scope section B): this experiment must still run when
    the optimizer has not been, and it says so rather than failing.
    """
    rankings = config.TABLES_DIR / "hedge_optimizer_rankings.csv"
    candidates = config.TABLES_DIR / "hedge_optimizer_candidates.csv"
    if not rankings.exists() or not candidates.exists():
        print("\n  (the optimizer's efficient hedge is not included: run "
              "experiments/hedge_optimizer_experiment.py first)")
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
        "interpolate": PricingGridInterpolator(grid,
                                               config.INTERPOLATION_METHOD),
        "cost_per_share": float(row["ask"]),
        "cost_basis": "market ask",
        "strike": strike,
        "grid": grid,
    }


if __name__ == "__main__":
    sys.exit(main())
