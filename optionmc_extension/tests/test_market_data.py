"""Market-data selection logic, tested offline.

These tests never touch the network. They drive the pure selection functions
with hand-built inputs, so the rules from scope section 6 -- expiry 60-90 days
out, strike at 95%-100% of spot, bid/ask midpoint with a last-price fallback --
are verified deterministically.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.market_data import (MarketDataError, estimate_dividend_yield,
                             parse_fred_csv, select_expiry,
                             select_put_contract)


AS_OF = date(2026, 1, 15)


def test_expiry_picks_the_middle_of_the_window():
    """With several candidates, the one nearest 75 days out wins."""
    expiries = ["2026-01-30", "2026-02-20", "2026-03-20", "2026-04-17",
                "2026-06-19"]
    expiry, days = select_expiry(expiries, AS_OF, 60, 90)
    assert expiry == "2026-03-20"          # 64 days
    assert days == 64


def test_expiry_ignores_dates_outside_the_window():
    expiries = ["2026-01-20", "2026-03-25", "2026-12-18"]
    expiry, days = select_expiry(expiries, AS_OF, 60, 90)
    assert expiry == "2026-03-25"
    assert 60 <= days <= 90


def test_expiry_warns_and_falls_back_when_window_is_empty():
    expiries = ["2026-01-20", "2026-12-18"]
    with pytest.warns(UserWarning, match="no expiry between"):
        expiry, days = select_expiry(expiries, AS_OF, 60, 90)
    assert expiry in expiries
    assert days > 0


def test_expiry_rejects_empty_and_past_only_input():
    with pytest.raises(MarketDataError):
        select_expiry([], AS_OF)
    with pytest.raises(MarketDataError):
        select_expiry(["2025-01-01"], AS_OF)


def _chain(strikes, bids=None, asks=None, last=None, iv=0.18):
    n = len(strikes)
    return pd.DataFrame({
        "strike": strikes,
        "bid": bids if bids is not None else [1.0] * n,
        "ask": asks if asks is not None else [1.2] * n,
        "lastPrice": last if last is not None else [1.1] * n,
        "impliedVolatility": [iv] * n,
    })


def test_strike_selected_in_the_required_moneyness_band():
    spot = 600.0
    puts = _chain([540.0, 570.0, 585.0, 595.0, 600.0, 620.0])
    contract = select_put_contract(puts, spot, 0.95, 1.00)
    # band is 570-600, centre 585
    assert contract["strike"] == 585.0
    assert 0.95 * spot <= contract["strike"] <= 1.00 * spot


def test_market_price_is_the_bid_ask_midpoint():
    puts = _chain([585.0], bids=[4.0], asks=[4.6])
    contract = select_put_contract(puts, 600.0, 0.95, 1.00)
    assert contract["market_price"] == pytest.approx(4.3)
    assert contract["price_source"] == "mid"


@pytest.mark.parametrize("bid,ask", [
    (0.0, 4.6),          # no bid
    (4.0, 0.0),          # no ask
    (np.nan, 4.6),       # missing bid
    (4.0, np.nan),       # missing ask
    (5.0, 4.0),          # crossed quote
])
def test_falls_back_to_last_price_when_quote_is_unusable(bid, ask):
    """Scope section 6 requires the fallback to be flagged, not hidden."""
    puts = _chain([585.0], bids=[bid], asks=[ask], last=[4.25])
    contract = select_put_contract(puts, 600.0, 0.95, 1.00)
    assert contract["market_price"] == pytest.approx(4.25)
    assert contract["price_source"] == "last"


def test_raises_when_neither_quote_nor_last_price_exists():
    puts = _chain([585.0], bids=[0.0], asks=[0.0], last=[0.0])
    with pytest.raises(MarketDataError, match="neither a usable bid/ask"):
        select_put_contract(puts, 600.0, 0.95, 1.00)


def test_warns_and_falls_back_when_no_strike_is_in_band():
    puts = _chain([400.0, 800.0])
    with pytest.warns(UserWarning, match="no put strike between"):
        contract = select_put_contract(puts, 600.0, 0.95, 1.00)
    assert contract["strike"] in (400.0, 800.0)


def test_rejects_empty_chain():
    with pytest.raises(MarketDataError):
        select_put_contract(pd.DataFrame(), 600.0)
    with pytest.raises(MarketDataError):
        select_put_contract(pd.DataFrame({"bid": [1.0]}), 600.0)


def test_dividend_yield_sums_the_trailing_year():
    index = pd.to_datetime(["2025-03-20", "2025-06-20", "2025-09-19",
                            "2025-12-19", "2026-01-15"])
    history = pd.DataFrame({"Dividends": [1.5, 1.6, 1.7, 1.8, 0.0]}, index=index)
    q = estimate_dividend_yield(history, spot=600.0)
    assert q == pytest.approx((1.5 + 1.6 + 1.7 + 1.8) / 600.0)


def test_dividend_yield_excludes_payments_older_than_the_lookback():
    index = pd.to_datetime(["2024-01-10", "2025-12-19", "2026-01-15"])
    history = pd.DataFrame({"Dividends": [99.0, 1.8, 0.0]}, index=index)
    q = estimate_dividend_yield(history, spot=600.0)
    assert q == pytest.approx(1.8 / 600.0)


def test_dividend_yield_is_zero_without_dividend_data():
    assert estimate_dividend_yield(pd.DataFrame({"Close": [1.0]}), 100.0) == 0.0
    index = pd.to_datetime(["2026-01-15"])
    empty = pd.DataFrame({"Dividends": [0.0]}, index=index)
    assert estimate_dividend_yield(empty, 100.0) == 0.0


def test_fred_parser_takes_the_last_valid_observation():
    """FRED writes missing values as a dot and quotes the rate in percent."""
    payload = ("observation_date,DGS3MO\n"
               "2026-01-12,4.28\n"
               "2026-01-13,4.31\n"
               "2026-01-14,.\n")
    date_str, rate = parse_fred_csv(payload, "DGS3MO")
    assert date_str == "2026-01-13"
    assert rate == pytest.approx(0.0431)


def test_fred_parser_handles_the_older_column_name():
    payload = "DATE,DGS3MO\n2026-01-13,4.31\n"
    date_str, rate = parse_fred_csv(payload, "DGS3MO")
    assert date_str == "2026-01-13"
    assert rate == pytest.approx(0.0431)


def test_fred_parser_rejects_useless_payloads():
    with pytest.raises(MarketDataError):
        parse_fred_csv("observation_date,DGS3MO\n2026-01-14,.\n", "DGS3MO")
    with pytest.raises(MarketDataError):
        parse_fred_csv("only_one_column\n1\n", "DGS3MO")
