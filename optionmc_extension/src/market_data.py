"""Real SPY market data: price history, option chain, and the risk-free rate.

Scope section 6. Everything downloaded is cached to CSV under data/, so a rerun
reproduces the same numbers instead of silently picking up a moved market.

SPY is a deliberate choice: its ETF options are American-style, so the early
exercise feature the LSMC extension exists to price is genuinely there. SPX and
XSP are European-style and would defeat the purpose.

The functions that make the *decisions* -- which expiry, which strike, which
quote -- are pure functions over plain data structures, so they are unit-tested
offline without touching the network.
"""
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .gbm import estimate_gbm_parameters


class MarketDataError(RuntimeError):
    """Raised when required market data is neither downloadable nor cached."""


@dataclass
class MarketSnapshot:
    """The frozen set of market inputs every later phase reads from."""
    as_of: str
    ticker: str
    spot: float                 # S0
    strike: float               # K
    expiry: str
    days_to_expiry: int
    time_to_expiry: float       # T in years
    market_put_price: float     # (bid + ask) / 2, or last price
    price_source: str           # "mid" or "last"
    bid: float
    ask: float
    implied_volatility: float
    historical_volatility: float    # sigma, annualised
    historical_drift: float         # mu, annualised, REAL-WORLD only
    dividend_yield: float           # q
    risk_free_rate: float           # r, continuously compounded
    risk_free_quoted: float         # as published by FRED, decimal
    risk_free_date: str
    history_start: str
    history_end: str
    n_history_days: int

    def to_json(self, path):
        path = str(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path):
        with open(str(path), "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


# --------------------------------------------------------------------------
# Pure selection logic (scope section 6) -- no network, fully unit-tested
# --------------------------------------------------------------------------

def select_expiry(expiries, as_of, min_days=60, max_days=90):
    """Pick the listed expiry roughly 60-90 days out.

    Parameters
    ----------
    expiries : sequence of str
        Expiry dates as "YYYY-MM-DD", as yfinance reports them.
    as_of : datetime.date
        Valuation date.
    min_days, max_days : int
        Inclusive window in calendar days.

    Returns
    -------
    (expiry_string, days_to_expiry)
        The expiry closest to the middle of the window. If none falls inside,
        the nearest expiry to the window centre is returned instead and a
        warning is issued, so the choice is never silently wrong.
    """
    if not len(expiries):
        raise MarketDataError("no expiries available")

    target = (min_days + max_days) / 2.0
    dated = []
    for e in expiries:
        days = (datetime.strptime(e, "%Y-%m-%d").date() - as_of).days
        if days > 0:
            dated.append((e, days))
    if not dated:
        raise MarketDataError("no expiries in the future")

    inside = [(e, d) for e, d in dated if min_days <= d <= max_days]
    pool = inside or dated
    if not inside:
        warnings.warn(
            f"no expiry between {min_days} and {max_days} days; "
            "falling back to the nearest listed expiry",
            stacklevel=2,
        )
    return min(pool, key=lambda item: abs(item[1] - target))


def select_put_contract(puts, spot, min_moneyness=0.95, max_moneyness=1.00):
    """Pick the put strike near 95%-100% of spot and read its quote.

    Parameters
    ----------
    puts : pandas.DataFrame
        Option-chain puts, with at least a "strike" column and preferably
        "bid", "ask", "lastPrice" and "impliedVolatility".
    spot : float
        Current underlying price S0.

    Returns
    -------
    dict
        strike, bid, ask, last_price, implied_volatility, market_price and
        price_source. `price_source` is "mid" when a usable bid/ask pair
        exists, otherwise "last" -- the scope requires the fallback to be
        reported rather than hidden.
    """
    if puts is None or len(puts) == 0:
        raise MarketDataError("empty put chain")
    if "strike" not in puts.columns:
        raise MarketDataError("put chain has no strike column")

    lo, hi = min_moneyness * spot, max_moneyness * spot
    target = 0.5 * (lo + hi)

    in_band = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
    if len(in_band) == 0:
        warnings.warn(
            f"no put strike between {lo:.2f} and {hi:.2f}; "
            "falling back to the nearest listed strike",
            stacklevel=2,
        )
        in_band = puts

    row = in_band.iloc[(in_band["strike"] - target).abs().argmin()]

    def _value(name):
        raw = row.get(name, np.nan)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")

    bid, ask = _value("bid"), _value("ask")
    last = _value("lastPrice")

    usable_quote = (
        np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid
    )
    if usable_quote:
        market_price, price_source = 0.5 * (bid + ask), "mid"
    elif np.isfinite(last) and last > 0:
        market_price, price_source = last, "last"
    else:
        raise MarketDataError(
            f"strike {float(row['strike'])} has neither a usable bid/ask nor a last price"
        )

    return {
        "strike": float(row["strike"]),
        "bid": bid,
        "ask": ask,
        "last_price": last,
        "implied_volatility": _value("impliedVolatility"),
        "market_price": float(market_price),
        "price_source": price_source,
    }


def estimate_dividend_yield(history, spot, lookback_days=365):
    """Trailing dividend yield q = (dividends paid over the last year) / spot.

    Needs a "Dividends" column, which `Ticker.history` supplies. Returns 0.0
    when no dividend information is present, which is the correct behaviour for
    a non-dividend-paying underlying.
    """
    if "Dividends" not in history.columns:
        return 0.0
    dividends = history["Dividends"]
    dividends = dividends[dividends > 0]
    if dividends.empty:
        return 0.0

    index = pd.to_datetime(dividends.index)
    cutoff = index.max() - pd.Timedelta(days=lookback_days)
    recent = dividends[index > cutoff]
    if recent.empty:
        return 0.0
    return float(recent.sum() / spot)


def parse_fred_csv(text, series=None):
    """Parse a FRED fredgraph.csv payload into (date, rate as a decimal).

    FRED writes missing observations as "." and quotes the rate in percent.
    Column naming has changed over time ("DATE" vs "observation_date"), so the
    date column is taken positionally.
    """
    from io import StringIO

    frame = pd.read_csv(StringIO(text))
    if frame.shape[1] < 2:
        raise MarketDataError("unexpected FRED payload")

    date_col = frame.columns[0]
    value_col = series if series in frame.columns else frame.columns[1]

    values = pd.to_numeric(frame[value_col], errors="coerce")
    valid = frame.loc[values.notna()]
    if valid.empty:
        raise MarketDataError("FRED series has no valid observations")

    last = valid.iloc[-1]
    return str(last[date_col]), float(values.loc[valid.index[-1]]) / 100.0


# --------------------------------------------------------------------------
# Network access, always with a CSV cache behind it
# --------------------------------------------------------------------------

def download_spy_history(ticker, period, cache_path, force_refresh=False):
    """Fetch daily history and cache it; fall back to the cache when offline.

    Raw (unadjusted) closes are kept, because the portfolio is valued as
    100 * S_h with no dividend income, and the dividend stream is handled
    separately through the yield q in the risk-neutral drift.
    """
    import os

    if not force_refresh and os.path.exists(cache_path):
        return load_spy_history(cache_path), "cache"

    try:
        import yfinance as yf

        frame = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if frame is None or frame.empty:
            raise MarketDataError(f"{ticker} history came back empty")
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.index.name = "Date"
        frame.to_csv(cache_path)
        return frame, "download"
    except MarketDataError:
        raise
    except Exception as exc:                       # network, parsing, API drift
        if os.path.exists(cache_path):
            warnings.warn(f"download failed ({exc}); using cached history",
                          stacklevel=2)
            return load_spy_history(cache_path), "cache"
        raise MarketDataError(
            f"could not download {ticker} history and no cache exists at "
            f"{cache_path}: {exc}"
        ) from exc


def load_spy_history(cache_path):
    """Read a cached history CSV back into a DataFrame."""
    frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    if "Close" not in frame.columns:
        raise MarketDataError(f"{cache_path} has no Close column")
    return frame


def download_risk_free_rate(url, cache_path, series="DGS3MO", force_refresh=False):
    """Fetch the FRED 3-month Treasury constant maturity rate, with caching.

    Returns
    -------
    dict with date, quoted (decimal, as published) and continuous
    (continuously compounded, r = ln(1 + quoted)) -- the convention every
    pricing formula in this project uses.
    """
    import os
    from urllib.request import urlopen

    text = None
    source = "download"
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        source = "cache"
    else:
        try:
            with urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8")
            with open(cache_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception as exc:
            if os.path.exists(cache_path):
                warnings.warn(f"FRED download failed ({exc}); using cache",
                              stacklevel=2)
                with open(cache_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                source = "cache"
            else:
                raise MarketDataError(
                    f"could not download {series} and no cache exists at "
                    f"{cache_path}: {exc}"
                ) from exc

    date, quoted = parse_fred_csv(text, series)
    return {
        "date": date,
        "quoted": quoted,
        "continuous": float(np.log1p(quoted)),
        "source": source,
    }


def download_option_snapshot(ticker, spot, as_of, cache_path,
                             min_days=60, max_days=90,
                             min_moneyness=0.95, max_moneyness=1.00,
                             force_refresh=False):
    """Select the SPY put required by the scope and cache the full chain slice.

    Returns
    -------
    (contract_dict, expiry_string, days_to_expiry, source)
    """
    import os

    if not force_refresh and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        expiry = str(cached["expiry"].iloc[0])
        days = int((datetime.strptime(expiry, "%Y-%m-%d").date() - as_of).days)
        contract = select_put_contract(cached, spot, min_moneyness, max_moneyness)
        return contract, expiry, days, "cache"

    try:
        import yfinance as yf

        handle = yf.Ticker(ticker)
        expiry, days = select_expiry(handle.options, as_of, min_days, max_days)
        puts = handle.option_chain(expiry).puts.copy()
        if puts.empty:
            raise MarketDataError(f"{ticker} {expiry} put chain is empty")
        puts["expiry"] = expiry
        puts.to_csv(cache_path, index=False)
        contract = select_put_contract(puts, spot, min_moneyness, max_moneyness)
        return contract, expiry, days, "download"
    except MarketDataError:
        raise
    except Exception as exc:
        if os.path.exists(cache_path):
            warnings.warn(f"option download failed ({exc}); using cache",
                          stacklevel=2)
            cached = pd.read_csv(cache_path)
            expiry = str(cached["expiry"].iloc[0])
            days = int((datetime.strptime(expiry, "%Y-%m-%d").date() - as_of).days)
            contract = select_put_contract(cached, spot, min_moneyness,
                                           max_moneyness)
            return contract, expiry, days, "cache"
        raise MarketDataError(
            f"could not download the {ticker} option chain and no cache exists "
            f"at {cache_path}: {exc}"
        ) from exc


def build_market_snapshot(cfg, force_refresh=False, trading_days_per_year=252):
    """Assemble every market input the later phases need, and cache it.

    `cfg` is the project `config` module. Returns a MarketSnapshot; the caller
    is expected to persist it so the whole study is reproducible from data/.
    """
    history, hist_source = download_spy_history(
        cfg.TICKER, cfg.HISTORY_PERIOD, cfg.SPY_HISTORY_CSV, force_refresh)

    closes = history["Close"].dropna()
    if closes.size < 30:
        raise MarketDataError("not enough history to estimate parameters")

    spot = float(closes.iloc[-1])
    as_of = closes.index[-1].date()

    params = estimate_gbm_parameters(closes.to_numpy(), trading_days_per_year)
    dividend_yield = estimate_dividend_yield(history, spot)

    rate = download_risk_free_rate(cfg.FRED_CSV_URL, cfg.RISK_FREE_CSV,
                                   cfg.FRED_SERIES, force_refresh)

    contract, expiry, days, _ = download_option_snapshot(
        cfg.TICKER, spot, as_of, cfg.SPY_OPTION_CSV,
        cfg.EXPIRY_MIN_DAYS, cfg.EXPIRY_MAX_DAYS,
        cfg.STRIKE_MIN_MONEYNESS, cfg.STRIKE_MAX_MONEYNESS,
        force_refresh)

    return MarketSnapshot(
        as_of=str(as_of),
        ticker=cfg.TICKER,
        spot=spot,
        strike=contract["strike"],
        expiry=expiry,
        days_to_expiry=int(days),
        time_to_expiry=days / 365.0,
        market_put_price=contract["market_price"],
        price_source=contract["price_source"],
        bid=contract["bid"],
        ask=contract["ask"],
        implied_volatility=contract["implied_volatility"],
        historical_volatility=params["sigma"],
        historical_drift=params["mu"],
        dividend_yield=dividend_yield,
        risk_free_rate=rate["continuous"],
        risk_free_quoted=rate["quoted"],
        risk_free_date=rate["date"],
        history_start=str(closes.index[0].date()),
        history_end=str(as_of),
        n_history_days=int(closes.size),
    )
