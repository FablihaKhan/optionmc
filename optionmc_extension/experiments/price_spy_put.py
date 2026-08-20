#!/usr/bin/env python
"""PHASE 5: price the selected real SPY American put.

Prices the contract chosen in data/market_snapshot.json with

  * Least-Squares Monte Carlo  (the extension's method)
  * a Cox-Ross-Rubinstein tree (the numerical benchmark, scope section 9)
  * Black-Scholes European     (to isolate the early-exercise premium)

and compares all of them with the observed market price. Runs under both the
historical and the implied volatility, because the choice materially changes
the answer and hiding it would be misleading.

    python experiments/price_spy_put.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.binomial import (crr_american_put, crr_european_put,
                          implied_volatility_american_put)
from src.black_scholes import bs_put, implied_volatility_put
from src.lsmc import price_american_put_lsmc
from src.market_data import MarketSnapshot
from src.sanity import check_american_put, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
OUTPUT_CSV = config.TABLES_DIR / "spy_american_put_pricing.csv"


def price_with(snapshot, sigma, label):
    """Price the contract at one volatility and return a result row."""
    common = dict(S0=snapshot.spot, K=snapshot.strike,
                  T=snapshot.time_to_expiry, r=snapshot.risk_free_rate,
                  sigma=sigma, q=snapshot.dividend_yield)

    t0 = time.perf_counter()
    lsmc = price_american_put_lsmc(
        n_paths=config.LSMC_N_PATHS, n_steps=config.LSMC_N_STEPS,
        degree=config.LSMC_DEGREE, seed=config.SEED,
        antithetic=config.LSMC_ANTITHETIC, **common)
    lsmc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    binomial = crr_american_put(n_steps=config.BINOMIAL_N_STEPS, **common)
    binomial_time = time.perf_counter() - t0

    european_tree = crr_european_put(n_steps=config.BINOMIAL_N_STEPS, **common)
    european_bs = bs_put(**common)

    absolute_error = abs(lsmc.price - binomial)
    return {
        "volatility_label": label,
        "sigma": sigma,
        "lsmc_price": lsmc.price,
        "lsmc_std_error": lsmc.std_error,
        "binomial_price": binomial,
        "market_price": snapshot.market_put_price,
        "european_bs_price": european_bs,
        "european_tree_price": european_tree,
        "early_exercise_premium": binomial - european_tree,
        "absolute_error_vs_binomial": absolute_error,
        "relative_error_vs_binomial": absolute_error / binomial,
        "absolute_error_vs_market": abs(lsmc.price - snapshot.market_put_price),
        "relative_error_vs_market": (abs(lsmc.price - snapshot.market_put_price)
                                     / snapshot.market_put_price),
        "early_exercise_fraction": lsmc.early_exercise_fraction,
        "lsmc_runtime_sec": lsmc_time,
        "binomial_runtime_sec": binomial_time,
        "_lsmc": lsmc,
        "_european_bs": european_bs,
    }


def main():
    if not SNAPSHOT_JSON.exists():
        print(f"No market snapshot at {SNAPSHOT_JSON}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snapshot = MarketSnapshot.from_json(SNAPSHOT_JSON)

    print("=" * 74)
    print(f"SPY AMERICAN PUT  K={snapshot.strike}  exp {snapshot.expiry}"
          f"  (as of {snapshot.as_of})")
    print("=" * 74)
    print(f"  S0 = {snapshot.spot:.2f}   T = {snapshot.time_to_expiry:.4f} yr"
          f"   r = {snapshot.risk_free_rate:.4f}   q = {snapshot.dividend_yield:.4f}")
    print(f"  LSMC: {config.LSMC_N_PATHS:,} paths x {config.LSMC_N_STEPS} steps,"
          f" degree {config.LSMC_DEGREE}, antithetic={config.LSMC_ANTITHETIC},"
          f" seed={config.SEED}")
    print(f"  Binomial benchmark: {config.BINOMIAL_N_STEPS:,} CRR steps")

    # The volatility field that comes with a downloaded option chain is often
    # stale and inconsistent with the quoted bid/ask, so we invert the observed
    # market price ourselves -- against the AMERICAN tree, because the contract
    # is American.
    american_iv = implied_volatility_american_put(
        snapshot.market_put_price, snapshot.spot, snapshot.strike,
        snapshot.time_to_expiry, snapshot.risk_free_rate,
        snapshot.dividend_yield, n_steps=500)
    european_iv = implied_volatility_put(
        snapshot.market_put_price, snapshot.spot, snapshot.strike,
        snapshot.time_to_expiry, snapshot.risk_free_rate,
        snapshot.dividend_yield)

    print("\n  implied volatility of the market price:")
    print(f"    from the chain feed        {snapshot.implied_volatility:.4f}"
          f"   (reprices to {bs_put(snapshot.spot, snapshot.strike, snapshot.time_to_expiry, snapshot.risk_free_rate, snapshot.implied_volatility, snapshot.dividend_yield):.4f},"
          f" market {snapshot.market_put_price:.4f})")
    print(f"    our European inversion     {european_iv:.4f}")
    print(f"    our American inversion     {american_iv:.4f}   <- used below")

    rows = [
        price_with(snapshot, snapshot.historical_volatility, "historical"),
        price_with(snapshot, american_iv, "market-implied"),
    ]

    all_passed = True
    for row in rows:
        print("\n" + "-" * 74)
        print(f"  volatility: {row['volatility_label']}  sigma = {row['sigma']:.4f}")
        print("-" * 74)
        print(f"    LSMC price        = {row['lsmc_price']:10.4f}"
              f"   (s.e. {row['lsmc_std_error']:.4f})")
        print(f"    Binomial price    = {row['binomial_price']:10.4f}")
        print(f"    Market price      = {row['market_price']:10.4f}"
              f"   ({snapshot.price_source})")
        print(f"    European (BS)     = {row['european_bs_price']:10.4f}")
        print(f"    Early ex. premium = {row['early_exercise_premium']:10.4f}"
              f"   ({row['early_exercise_fraction']:.1%} of paths exercise early)")
        print(f"    Absolute error    = {row['absolute_error_vs_binomial']:10.4f}"
              f"   (LSMC vs binomial)")
        print(f"    Relative error    = {row['relative_error_vs_binomial']:10.4%}")
        print(f"    Runtime           = {row['lsmc_runtime_sec']:10.4f} s LSMC,"
              f" {row['binomial_runtime_sec']:.4f} s binomial")

        print("    sanity checks (scope section 10):")
        checks = check_american_put(row["lsmc_price"], row["_european_bs"],
                                    snapshot.spot, snapshot.strike)
        checks.append(
            type(checks[0])("LSMC within 3 s.e. of binomial",
                            row["absolute_error_vs_binomial"]
                            <= 3 * row["lsmc_std_error"] + 0.01,
                            f"|{row['absolute_error_vs_binomial']:.4f}| vs "
                            f"3 s.e. = {3 * row['lsmc_std_error']:.4f}")
        )
        all_passed &= report(checks, printer=lambda s: print("  " + s))

    frame = pd.DataFrame([{k: v for k, v in row.items()
                           if not k.startswith("_")} for row in rows])
    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  table saved to {OUTPUT_CSV}")

    print("\n  How to read this:")
    print("   * The HISTORICAL-volatility row is the economic comparison: the")
    print("     model price against what the market actually charges. Any gap")
    print("     is a volatility premium, not a defect in the pricer.")
    print("   * The MARKET-IMPLIED row is a consistency check. That sigma was")
    print("     solved to make the binomial reproduce the market price, so the")
    print("     binomial column must land on the market price by construction;")
    print("     what it really tests is whether LSMC agrees with the tree.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
