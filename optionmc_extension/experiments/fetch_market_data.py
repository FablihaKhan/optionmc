#!/usr/bin/env python
"""PHASE 5: download and cache the SPY market data the study runs on.

    python experiments/fetch_market_data.py            # use cache when present
    python experiments/fetch_market_data.py --refresh  # force a fresh download

Writes data/spy_history.csv, data/spy_option_snapshot.csv,
data/risk_free_dgs3mo.csv and data/market_snapshot.json. Every later phase
reads the snapshot rather than the network, so results stay reproducible.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.market_data import MarketDataError, build_market_snapshot

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and download fresh data")
    args = parser.parse_args()

    try:
        snapshot = build_market_snapshot(
            config, force_refresh=args.refresh,
            trading_days_per_year=config.TRADING_DAYS_PER_YEAR)
    except MarketDataError as exc:
        print("MARKET DATA UNAVAILABLE")
        print(f"  {exc}")
        print("\nRun this script once on a machine with internet access; the")
        print("CSV caches it writes under data/ are all the later phases need.")
        return 1

    snapshot.to_json(SNAPSHOT_JSON)

    print("=" * 70)
    print(f"MARKET SNAPSHOT  {snapshot.ticker}  as of {snapshot.as_of}")
    print("=" * 70)
    print(f"  spot S0                  {snapshot.spot:12.4f}")
    print(f"  strike K                 {snapshot.strike:12.4f}"
          f"   (moneyness {snapshot.strike / snapshot.spot:.4f})")
    print(f"  expiry                   {snapshot.expiry:>12}"
          f"   ({snapshot.days_to_expiry} days, T = {snapshot.time_to_expiry:.4f} yr)")
    print(f"  market put price         {snapshot.market_put_price:12.4f}"
          f"   (source: {snapshot.price_source}, bid {snapshot.bid}, ask {snapshot.ask})")
    print(f"  implied volatility       {snapshot.implied_volatility:12.4f}")
    print("-" * 70)
    print(f"  historical sigma         {snapshot.historical_volatility:12.4f}"
          f"   <- used for pricing AND risk")
    print(f"  historical mu            {snapshot.historical_drift:12.4f}"
          f"   <- REAL WORLD, risk simulation only")
    print(f"  dividend yield q         {snapshot.dividend_yield:12.4f}")
    print(f"  risk-free r (cont.)      {snapshot.risk_free_rate:12.4f}"
          f"   (FRED {config.FRED_SERIES} {snapshot.risk_free_quoted:.4%}"
          f" on {snapshot.risk_free_date})")
    print(f"  risk-neutral drift r-q   {snapshot.risk_free_rate - snapshot.dividend_yield:12.4f}"
          f"   <- pricing only")
    print("-" * 70)
    print(f"  history   {snapshot.history_start} .. {snapshot.history_end}"
          f"  ({snapshot.n_history_days} trading days)")
    print(f"  snapshot saved to {SNAPSHOT_JSON}")

    if snapshot.price_source == "last":
        print("\n  NOTE: no usable bid/ask; the market price is the last traded")
        print("        price. This is reported, per scope section 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
