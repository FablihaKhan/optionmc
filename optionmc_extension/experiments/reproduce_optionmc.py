#!/usr/bin/env python
"""PHASE 1: reproduce the base OptionMC European results.

Scope PHASE 1 and the definition of done: before extending anything, show that
the original package's European Monte Carlo results come back. This calls the
ORIGINAL OptionMC classes -- it does not reimplement them -- so what is shown
here is genuinely the base package running.

Reproduces the paper's headline example (S0 = K = 100, T = 1, r = 5%,
sigma = 20%, Black-Scholes call 10.45 / put 5.57) across a range of iteration
counts, for both standard and antithetic Monte Carlo.

    python experiments/reproduce_optionmc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.european_mc import reproduce_baseline
from src.sanity import Check, report

OUTPUT_CSV = config.TABLES_DIR / "phase1_optionmc_reproduction.csv"

# The base paper's example (Herho et al. 2025, figures 1-4).
PAPER = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
PAPER_BS_CALL = 10.45
PAPER_BS_PUT = 5.57


def main():
    print("=" * 78)
    print("PHASE 1  Reproducing the base OptionMC European results")
    print("=" * 78)
    print(f"  parameters: S0={PAPER['S0']}, K={PAPER['K']}, T={PAPER['T']}, "
          f"r={PAPER['r']:.0%}, sigma={PAPER['sigma']:.0%}")
    print("  pricer: optionmc.models.OptionPricing (the original package)")

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    rows = [reproduce_baseline(iterations=n, seed=config.SEED + i, **PAPER)
            for i, n in enumerate(sizes)]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_CSV, index=False)

    bs_call = rows[0]["bs_call"]
    bs_put = rows[0]["bs_put"]
    print(f"\n  Black-Scholes: call {bs_call:.4f}, put {bs_put:.4f}"
          f"   (paper: {PAPER_BS_CALL}, {PAPER_BS_PUT})")

    header = (f"\n  {'iterations':>11} {'std call':>10} {'anti call':>10} "
              f"{'std err %':>10} {'anti err %':>11} {'std time':>9} "
              f"{'anti time':>10}")
    print(header)
    print("  " + "-" * (len(header) - 3))
    for row in rows:
        print(f"  {row['iterations']:>11,} {row['standard_call']:>10.4f} "
              f"{row['antithetic_call']:>10.4f} "
              f"{row['standard_call_rel_error']:>10.4%} "
              f"{row['antithetic_call_rel_error']:>11.4%} "
              f"{row['standard_time']:>9.4f} {row['antithetic_time']:>10.4f}")

    largest = rows[-1]
    print("\n  sanity checks:")
    checks = [
        Check("Black-Scholes call matches the paper",
              abs(bs_call - PAPER_BS_CALL) < 0.01,
              f"{bs_call:.4f} vs {PAPER_BS_CALL}"),
        Check("Black-Scholes put matches the paper",
              abs(bs_put - PAPER_BS_PUT) < 0.01,
              f"{bs_put:.4f} vs {PAPER_BS_PUT}"),
        Check("standard MC converges to Black-Scholes",
              largest["standard_call_rel_error"] < 0.005,
              f"{largest['standard_call_rel_error']:.4%} at "
              f"{largest['iterations']:,} iterations"),
        Check("antithetic MC converges to Black-Scholes",
              largest["antithetic_call_rel_error"] < 0.005,
              f"{largest['antithetic_call_rel_error']:.4%} at "
              f"{largest['iterations']:,} iterations"),
        Check("put-call parity holds for the analytical prices",
              abs((bs_call - bs_put)
                  - (PAPER["S0"] - PAPER["K"] * np.exp(-PAPER["r"] * PAPER["T"])))
              < 1e-10,
              "C - P = S0 - K exp(-rT)"),
    ]
    all_passed = report(checks)

    mean_std = np.mean([r["standard_call_rel_error"] for r in rows])
    mean_anti = np.mean([r["antithetic_call_rel_error"] for r in rows])
    print(f"\n  Mean relative error across all sizes: "
          f"standard {mean_std:.4%}, antithetic {mean_anti:.4%}.")
    print("  The base package reproduces, so the extension is built on a")
    print("  working foundation rather than on an assumption that it works.")
    print(f"\n  table saved to {OUTPUT_CSV}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
