#!/usr/bin/env python
"""PHASE 6, Experiment 2: how the number of time steps affects the LSMC price.

Scope section 11. Steps = 10, 25, 50, 100; report price, error and runtime.

The error is split into its two genuinely different sources, because lumping
them together would hide what is actually happening:

    LSMC(m) vs Bermudan(m)    Monte Carlo error -- the simulation's own noise
                              and bias at a fixed exercise grid
    Bermudan(m) vs American   discretisation error -- the value given up by
                              only being allowed to exercise m times

More steps shrink the second and cost runtime; they do not shrink the first.

    python experiments/discretization_test.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.binomial import crr_american_put, crr_bermudan_put
from src.market_data import MarketSnapshot
from src.replication import replicate_lsmc, summarise
from src.sanity import Check, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
OUTPUT_CSV = config.TABLES_DIR / "experiment2_discretization.csv"

# Tree resolution for the Bermudan benchmarks: a common multiple of every
# step count under test, so each exercise date lands exactly on a tree node.
BERMUDAN_TREE_STEPS = 2000


def main():
    if not SNAPSHOT_JSON.exists():
        print(f"No market snapshot at {SNAPSHOT_JSON}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    contract = dict(S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
                    r=snap.risk_free_rate, sigma=snap.historical_volatility,
                    q=snap.dividend_yield)
    american = crr_american_put(n_steps=config.BINOMIAL_N_STEPS, **contract)

    print("=" * 78)
    print("EXPERIMENT 2  Time discretisation: number of exercise dates")
    print("=" * 78)
    print(f"  contract: SPY American put, K={snap.strike}, exp {snap.expiry}, "
          f"S0={snap.spot:.2f}, T={snap.time_to_expiry:.4f}")
    print(f"  continuously exercisable American benchmark = {american:.6f}")
    print(f"  {config.N_REPLICATIONS} replications per step count, "
          f"{config.LSMC_N_PATHS:,} paths, degree {config.LSMC_DEGREE}")
    print()

    rows = []
    for n_steps in config.DISCRETIZATION_STEPS:
        bermudan = crr_bermudan_put(n_steps=BERMUDAN_TREE_STEPS,
                                    n_exercise_dates=n_steps, **contract)
        runs = replicate_lsmc(
            config.N_REPLICATIONS, config.SEED,
            n_paths=config.LSMC_N_PATHS, n_steps=n_steps,
            degree=config.LSMC_DEGREE, antithetic=config.LSMC_ANTITHETIC,
            **contract)

        row = summarise(runs["prices"], bermudan, runs["runtimes"],
                        runs["reported_std_errors"])
        row["n_steps"] = n_steps
        row["bermudan_benchmark"] = bermudan
        row["american_benchmark"] = american
        row["discretisation_error"] = american - bermudan
        row["mc_error_vs_bermudan"] = abs(row["mean_price"] - bermudan)
        row["total_error_vs_american"] = abs(row["mean_price"] - american)
        row["mean_early_exercise_fraction"] = float(
            runs["early_exercise_fractions"].mean())
        rows.append(row)

    frame = pd.DataFrame(rows)[[
        "n_steps", "n_replications", "mean_price", "std_price",
        "bermudan_benchmark", "american_benchmark", "mc_error_vs_bermudan",
        "discretisation_error", "total_error_vs_american", "rmse",
        "mean_runtime_sec", "mean_early_exercise_fraction",
    ]]
    frame.to_csv(OUTPUT_CSV, index=False)

    header = (f"{'steps':>6} {'LSMC price':>11} {'std dev':>8} "
              f"{'Bermudan':>10} {'MC err':>9} {'discr err':>10} "
              f"{'total err':>10} {'runtime s':>10}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['n_steps']:>6} {row['mean_price']:>11.4f} "
              f"{row['std_price']:>8.4f} {row['bermudan_benchmark']:>10.4f} "
              f"{row['mc_error_vs_bermudan']:>9.4f} "
              f"{row['discretisation_error']:>10.4f} "
              f"{row['total_error_vs_american']:>10.4f} "
              f"{row['mean_runtime_sec']:>10.4f}")

    print("\n  sanity checks (scope section 10):")
    bermudans = [r["bermudan_benchmark"] for r in rows]
    discr = [r["discretisation_error"] for r in rows]
    runtimes = [r["mean_runtime_sec"] for r in rows]
    checks = [
        Check("Bermudan value rises with more exercise dates",
              all(a <= b + 1e-12 for a, b in zip(bermudans, bermudans[1:])),
              " < ".join(f"{b:.4f}" for b in bermudans)),
        Check("discretisation error shrinks with more steps",
              all(a >= b - 1e-12 for a, b in zip(discr, discr[1:])),
              " > ".join(f"{d:.4f}" for d in discr)),
        Check("Bermudan never exceeds the American value",
              all(b <= american + 1e-12 for b in bermudans),
              f"max {max(bermudans):.4f} <= {american:.4f}"),
        Check("runtime grows with more steps",
              all(a <= b * 1.15 for a, b in zip(runtimes, runtimes[1:])),
              " -> ".join(f"{t:.4f}s" for t in runtimes)),
    ]
    all_passed = report(checks)

    first, last = rows[0], rows[-1]
    print(f"\n  Going from {first['n_steps']} to {last['n_steps']} exercise dates buys")
    print(f"  {first['discretisation_error'] - last['discretisation_error']:.4f} "
          f"of option value and costs "
          f"{last['mean_runtime_sec'] / first['mean_runtime_sec']:.1f}x the runtime.")
    print(f"  Monte Carlo error stays around "
          f"{sum(r['mc_error_vs_bermudan'] for r in rows) / len(rows):.4f} "
          f"throughout: more steps do not fix it, only more paths do.")

    best_total = min(rows, key=lambda r: r["total_error_vs_american"])
    if best_total["n_steps"] != rows[-1]["n_steps"]:
        print(f"\n  CAUTION: the smallest total error is at "
              f"{best_total['n_steps']} steps, but that is not the most")
        print("  accurate setting. At a fixed path count the LSMC sits ABOVE its")
        print("  own Bermudan benchmark while the coarse exercise grid pulls the")
        print("  true value DOWN, so the two errors cancel by coincidence. Judge")
        print("  accuracy by the two columns separately, never by their sum.")
    print(f"\n  table saved to {OUTPUT_CSV}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
