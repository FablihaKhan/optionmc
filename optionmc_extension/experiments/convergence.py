#!/usr/bin/env python
"""PHASE 6, Experiment 1: LSMC convergence in the number of paths.

Scope section 11. For N = 1000, 5000, 10000, 25000, 50000 measure the LSMC
price, its absolute and relative error against the binomial benchmark, the
runtime and the standard error -- each averaged over independent replications,
so the numbers describe the estimator rather than one lucky seed.

The fitted convergence order should come out near -1/2: Monte Carlo error falls
as 1/sqrt(N), which is the same rate the base OptionMC paper reports for its
European estimates.

    python experiments/convergence.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.binomial import crr_american_put
from src.market_data import MarketSnapshot
from src.replication import fit_convergence_order, replicate_lsmc, summarise
from src.sanity import check_convergence, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
OUTPUT_CSV = config.TABLES_DIR / "experiment1_convergence.csv"


def main():
    if not SNAPSHOT_JSON.exists():
        print(f"No market snapshot at {SNAPSHOT_JSON}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    contract = dict(S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
                    r=snap.risk_free_rate, sigma=snap.historical_volatility,
                    q=snap.dividend_yield)
    benchmark = crr_american_put(n_steps=config.BINOMIAL_N_STEPS, **contract)

    print("=" * 78)
    print("EXPERIMENT 1  Monte Carlo convergence in the number of paths")
    print("=" * 78)
    print(f"  contract: SPY American put, K={snap.strike}, exp {snap.expiry}, "
          f"S0={snap.spot:.2f}, T={snap.time_to_expiry:.4f}")
    print(f"  sigma={contract['sigma']:.4f}  r={contract['r']:.4f}  "
          f"q={contract['q']:.4f}")
    print(f"  binomial benchmark ({config.BINOMIAL_N_STEPS} steps) = "
          f"{benchmark:.6f}")
    print(f"  {config.N_REPLICATIONS} independent replications per path count, "
          f"{config.LSMC_N_STEPS} time steps, degree {config.LSMC_DEGREE}, "
          f"antithetic={config.LSMC_ANTITHETIC}")
    print()

    rows = []
    for n_paths in config.CONVERGENCE_PATHS:
        runs = replicate_lsmc(
            config.N_REPLICATIONS, config.SEED, n_paths=n_paths,
            n_steps=config.LSMC_N_STEPS, degree=config.LSMC_DEGREE,
            antithetic=config.LSMC_ANTITHETIC, **contract)
        row = summarise(runs["prices"], benchmark, runs["runtimes"],
                        runs["reported_std_errors"])
        row["n_paths"] = n_paths
        row["mean_early_exercise_fraction"] = float(
            runs["early_exercise_fractions"].mean())
        rows.append(row)

    frame = pd.DataFrame(rows)[[
        "n_paths", "n_replications", "mean_price", "std_price",
        "mean_reported_std_error", "bias", "mean_absolute_error", "rmse",
        "relative_rmse", "mean_runtime_sec", "mean_early_exercise_fraction",
        "benchmark",
    ]]
    frame.to_csv(OUTPUT_CSV, index=False)

    header = (f"{'N paths':>9} {'mean price':>11} {'std dev':>9} "
              f"{'bias':>9} {'abs err':>9} {'RMSE':>9} {'rel RMSE':>9} "
              f"{'runtime s':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['n_paths']:>9,} {row['mean_price']:>11.4f} "
              f"{row['std_price']:>9.4f} {row['bias']:>9.4f} "
              f"{row['mean_absolute_error']:>9.4f} {row['rmse']:>9.4f} "
              f"{row['relative_rmse']:>9.4%} {row['mean_runtime_sec']:>10.4f}")

    fit = fit_convergence_order([r["n_paths"] for r in rows],
                                [r["rmse"] for r in rows])
    print(f"\n  fitted convergence order: RMSE ~ N^({fit['order']:.3f})"
          f"   (theory: -0.500, R^2 = {fit['r_squared']:.4f})")

    sample_std_fit = fit_convergence_order([r["n_paths"] for r in rows],
                                           [r["std_price"] for r in rows])
    print(f"  spread of prices          : std  ~ N^({sample_std_fit['order']:.3f})"
          f"   (R^2 = {sample_std_fit['r_squared']:.4f})")

    print("\n  sanity checks (scope section 10):")
    checks = [
        check_convergence([r["mean_price"] for r in rows],
                          "price stabilises as N increases", tolerance=0.02),
        check_convergence([r["rmse"] for r in rows],
                          "RMSE keeps falling", tolerance=1.0),
    ]
    all_passed = report(checks)

    largest = rows[-1]
    print(f"\n  At N = {largest['n_paths']:,} the LSMC price is "
          f"{largest['mean_price']:.4f} against a benchmark of {benchmark:.4f}:")
    print(f"  a bias of {largest['bias']:+.4f} and a run-to-run spread of "
          f"{largest['std_price']:.4f}.")
    print(f"\n  table saved to {OUTPUT_CSV}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
