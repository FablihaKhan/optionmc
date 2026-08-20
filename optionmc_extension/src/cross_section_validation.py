"""Out-of-sample validation across a real SPY option cross-section.

The project's single-contract result answers "can the model reproduce this one
price". That is a weak question, and there is a way of answering it that looks
strong and means nothing: take a market price, solve for the volatility that
reproduces it, put that volatility back into the model, and report a tiny
error. The model has been handed the answer.

This module asks a harder question. For each expiry the usable strikes are
split deterministically into a calibration set and a held-out set. Only the
calibration contracts have their implied volatilities solved from the market.
A smile is fitted through those points alone, over log-moneyness. Each held-out
contract then reads its volatility off that curve -- from its *neighbours*, not
from itself -- and is priced with CRR and with LSMC. Only then is the result
compared with the held-out contract's own market price.

A held-out contract's own price never touches anything used to predict it. The
split forces the first and last strike of every expiry into the calibration
set, so every held-out point is interpolated between two calibration points
rather than extrapolated past the end of the curve.

American implied volatility comes from the CRR tree, never from Black-Scholes:
a European formula applied to an American quote absorbs the early-exercise
premium into sigma, and every price built on it inherits that error.
"""
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .binomial import crr_american_put, implied_volatility_american_put
from .lsmc import price_american_put_lsmc
from .market_data import MarketDataError, select_expiry

# Scope: three maturity groups. The nearest listed expiry to each target is
# used -- these are the dates the exchange actually lists.
MATURITY_GROUPS = (
    {"label": "short", "target_days": 30},
    {"label": "medium", "target_days": 60},
    {"label": "long", "target_days": 90},
)
EXPIRY_WINDOW_DAYS = 15

MIN_MONEYNESS = 0.90
MAX_MONEYNESS = 1.05

# Percentage errors are reported only for contracts worth at least this much.
# A put quoted at three cents can be mispriced by one cent and show a 33%
# "error" that says nothing about the model and everything about the divisor.
MIN_PRICE_FOR_PERCENTAGE = 0.10

SNAPSHOT_COLUMNS = [
    "timestamp", "ticker", "group", "expiry", "days_to_expiry",
    "time_to_expiry", "spot", "strike", "moneyness", "log_moneyness",
    "contract_symbol", "bid", "ask", "mid", "spread", "spread_percent",
    "volume", "open_interest", "quoted_iv",
]


# --------------------------------------------------------------------------
# Building the market universe
# --------------------------------------------------------------------------

def select_cross_section_expiries(expiries, as_of, groups=MATURITY_GROUPS,
                                  window_days=EXPIRY_WINDOW_DAYS):
    """Nearest listed expiry to each maturity target.

    Reuses the project's expiry chooser once per group. If two groups land on
    the same listed date -- which happens when the board is sparse -- the date
    appears once, so the cross-section never double-counts an expiry.
    """
    chosen, seen = [], set()
    for group in groups:
        target = group["target_days"]
        expiry, days = select_expiry(expiries, as_of,
                                     max(1, target - window_days),
                                     target + window_days)
        if expiry in seen:
            continue
        seen.add(expiry)
        chosen.append({"label": group["label"], "expiry": expiry,
                       "days_to_expiry": int(days),
                       "target_days": int(target)})
    if not chosen:
        raise MarketDataError("no expiry could be selected")
    return chosen


def build_contract_frame(puts, spot, expiry, days_to_expiry, group, timestamp,
                         ticker="SPY", min_moneyness=MIN_MONEYNESS,
                         max_moneyness=MAX_MONEYNESS):
    """One tidy row per listed put inside the moneyness band.

    Everything the later stages need is recorded here, including the quote
    fields that are only used for reporting. Nothing missing is invented: a
    field the feed does not supply stays NaN.
    """
    import pandas as pd

    if "strike" not in puts.columns:
        raise MarketDataError("put chain has no strike column")

    frame = puts.copy()
    frame["strike"] = frame["strike"].astype(float)
    lo, hi = min_moneyness * spot, max_moneyness * spot
    frame = frame[(frame["strike"] >= lo) & (frame["strike"] <= hi)].copy()

    def column(name):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").astype(float)
        return pd.Series(np.nan, index=frame.index, dtype=float)

    bid, ask = column("bid"), column("ask")
    mid = 0.5 * (bid + ask)

    out = pd.DataFrame({
        "timestamp": str(timestamp),
        "ticker": ticker,
        "group": group,
        "expiry": str(expiry),
        "days_to_expiry": int(days_to_expiry),
        "time_to_expiry": int(days_to_expiry) / 365.0,
        "spot": float(spot),
        "strike": frame["strike"].to_numpy(),
        "moneyness": frame["strike"].to_numpy() / spot,
        "log_moneyness": np.log(frame["strike"].to_numpy() / spot),
        "contract_symbol": (frame["contractSymbol"].astype(str)
                            if "contractSymbol" in frame.columns else ""),
        "bid": bid.to_numpy(),
        "ask": ask.to_numpy(),
        "mid": mid.to_numpy(),
        "spread": (ask - bid).to_numpy(),
        "spread_percent": np.where(mid.to_numpy() > 0,
                                   (ask - bid).to_numpy()
                                   / np.where(mid.to_numpy() > 0,
                                              mid.to_numpy(), np.nan) * 100.0,
                                   np.nan),
        "volume": column("volume").to_numpy(),
        "open_interest": column("openInterest").to_numpy(),
        "quoted_iv": column("impliedVolatility").to_numpy(),
    })
    return out.sort_values("strike").reset_index(drop=True)[SNAPSHOT_COLUMNS]


def download_cross_section(ticker, cache_path, groups=MATURITY_GROUPS,
                           min_moneyness=MIN_MONEYNESS,
                           max_moneyness=MAX_MONEYNESS, force_refresh=False,
                           as_of=None):
    """Fetch put chains for three maturities, or reload the cached snapshot.

    Written to its own file so the single-contract snapshot the rest of the
    project depends on is never touched.

    Returns
    -------
    (frame, source) with source "cache" or "download".
    """
    import os

    import pandas as pd

    if not force_refresh and os.path.exists(cache_path):
        return pd.read_csv(cache_path), "cache"

    try:
        import yfinance as yf

        handle = yf.Ticker(ticker)
        history = handle.history(period="5d")
        closes = history["Close"].dropna()
        if closes.empty:
            raise MarketDataError(f"no recent {ticker} price")
        spot = float(closes.iloc[-1])
        stamp = as_of or closes.index[-1].date()

        chosen = select_cross_section_expiries(handle.options, stamp, groups)
        frames = []
        for entry in chosen:
            puts = handle.option_chain(entry["expiry"]).puts
            if puts.empty:
                continue
            frames.append(build_contract_frame(
                puts, spot, entry["expiry"], entry["days_to_expiry"],
                entry["label"], stamp, ticker, min_moneyness, max_moneyness))
        if not frames:
            raise MarketDataError("every selected expiry returned an empty chain")

        frame = pd.concat(frames, ignore_index=True)

        # Outside trading hours the feed returns zero bid and zero ask for
        # every contract while still returning a last price. Caching that would
        # bury a snapshot with no usable quotes in the data directory, where a
        # later run would happily load it. Refuse instead, and say why.
        quoted = int(((frame["bid"] > 0) & (frame["ask"] > 0)).sum())
        if quoted == 0:
            raise MarketDataError(
                f"the {ticker} feed returned no live bid/ask on any of "
                f"{len(frame)} contracts, which is what it does when the "
                "market is closed. Nothing was cached; re-run during trading "
                "hours for a quoted cross-section.")

        frame.to_csv(cache_path, index=False)
        return frame, "download"

    except MarketDataError:
        raise
    except Exception as exc:
        if os.path.exists(cache_path):
            import warnings
            warnings.warn(f"cross-section download failed ({exc}); using cache",
                          stacklevel=2)
            return pd.read_csv(cache_path), "cache"
        raise MarketDataError(
            f"could not download the {ticker} cross-section and no cache "
            f"exists at {cache_path}: {exc}") from exc


def cross_section_from_cached_chain(chain, spot, expiry, days_to_expiry,
                                    timestamp, group="cached", ticker="SPY",
                                    min_moneyness=MIN_MONEYNESS,
                                    max_moneyness=MAX_MONEYNESS):
    """Build the cross-section frame from a single-expiry chain already on disk.

    The fallback for when the live feed has no quotes. It covers the strike
    dimension with genuinely quoted contracts rather than covering the maturity
    dimension with stale last-trade prices from a closed market, which would be
    non-synchronous across strikes and would show up as smile roughness that
    the held-out errors would then blame on the model.
    """
    return build_contract_frame(chain, spot, expiry, days_to_expiry, group,
                                timestamp, ticker, min_moneyness, max_moneyness)


def thin_by_spacing(frame, spacing, group_column="expiry"):
    """Keep strikes at least `spacing` apart, walking up from the lowest.

    SPY lists strikes a dollar apart near the money. Splitting that ladder into
    alternating calibration and test sets asks the smile to interpolate across
    one dollar, which any smooth curve does almost exactly -- the validation
    would pass without having tested anything. Thinning first puts a real gap
    between neighbouring calibration points.

    This is a design choice about strike density, not a quality filter, and the
    count it removes is reported by the caller rather than hidden.
    """
    if spacing is None or spacing <= 0:
        return frame.reset_index(drop=True)

    keep = []
    for _, block in frame.groupby(group_column):
        block = block.sort_values("strike")
        last = None
        for label, row in block.iterrows():
            if last is None or row["strike"] - last >= spacing - 1e-9:
                keep.append(label)
                last = row["strike"]
    return frame.loc[sorted(keep)].reset_index(drop=True)


def calibration_at_spacing(pool, spacing, group_column="expiry"):
    """Thin a calibration pool while keeping each expiry's end strikes.

    Used by the density study, where the held-out set is held fixed and only
    the calibration grid is thinned. Comparing spacings that each test a
    different set of contracts would confound the two, so the test set stays
    put and the endpoints are always retained -- otherwise a wide spacing would
    leave the outermost held-out strikes past the end of the smile and quietly
    drop them from the comparison.
    """
    if spacing is None or spacing <= 0:
        return pool.copy()

    keep = []
    for _, block in pool.groupby(group_column):
        block = block.sort_values("strike")
        labels = list(block.index)
        chosen, last = [], None
        for label in labels:
            strike = float(block.loc[label, "strike"])
            if last is None or strike - last >= spacing - 1e-9:
                chosen.append(label)
                last = strike
        if labels and labels[-1] not in chosen:
            chosen.append(labels[-1])
        keep.extend(chosen)
    return pool.loc[sorted(keep)].copy()


# --------------------------------------------------------------------------
# Quote quality
# --------------------------------------------------------------------------

@dataclass
class FilterReport:
    """What the quote filters removed, and why."""
    n_input: int
    n_kept: int
    reasons: dict = field(default_factory=dict)

    @property
    def n_removed(self):
        return self.n_input - self.n_kept

    def __str__(self):
        if not self.reasons:
            return f"{self.n_kept} of {self.n_input} contracts kept"
        detail = ", ".join(f"{count} {reason}"
                           for reason, count in sorted(self.reasons.items()))
        return (f"{self.n_kept} of {self.n_input} contracts kept; "
                f"removed {detail}")


def apply_quote_filters(frame, max_spread_percent=None, min_mid=0.0):
    """Drop contracts whose quotes cannot support a price comparison.

    Every rejection is counted by reason and returned, because a filter that
    silently removes half the sample can make any model look accurate.

    Parameters
    ----------
    max_spread_percent : float or None
        Optional cap on the bid-ask spread as a percentage of the mid. None
        keeps every contract that passes the basic validity rules.
    """
    frame = frame.copy()
    reasons = {}

    bid, ask, mid = frame["bid"], frame["ask"], frame["mid"]
    tests = {
        "missing bid or ask": ~(np.isfinite(bid) & np.isfinite(ask)),
        "crossed quote (ask < bid)": np.isfinite(bid) & np.isfinite(ask) & (ask < bid),
        "non-positive mid": np.isfinite(mid) & (mid <= min_mid),
    }
    if max_spread_percent is not None:
        tests[f"spread wider than {max_spread_percent:g}%"] = (
            np.isfinite(frame["spread_percent"])
            & (frame["spread_percent"] > max_spread_percent))

    drop = np.zeros(len(frame), dtype=bool)
    for reason, failed in tests.items():
        failed = np.asarray(failed, dtype=bool) & ~drop
        count = int(failed.sum())
        if count:
            reasons[reason] = count
        drop |= failed

    kept = frame[~drop].reset_index(drop=True)
    return kept, FilterReport(len(frame), len(kept), reasons)


# --------------------------------------------------------------------------
# Calibration / held-out split
# --------------------------------------------------------------------------

def split_calibration_test(frame, group_column="expiry"):
    """Alternate strikes into calibration and held-out sets, per expiry.

    Deterministic, so the split is the same on every run and nobody can tune it
    after seeing the errors. Strikes are sorted, even positions calibrate and
    odd positions are held out.

    The first and last strike of each expiry are forced into the calibration
    set. Without that, an even-length strike list would put its last strike --
    the furthest one out -- into the held-out set with no calibration point
    beyond it, and predicting it would mean extrapolating off the end of the
    smile rather than interpolating within it.

    Returns a copy with a `role` column of "calibration" or "test".
    """
    frame = frame.copy()
    frame["role"] = "calibration"

    for _, index in frame.groupby(group_column).groups.items():
        block = frame.loc[index].sort_values("strike")
        positions = list(block.index)
        n = len(positions)
        if n < 3:
            # Too few strikes to interpolate anything: keep them all for
            # calibration rather than manufacture a test point.
            continue
        for offset, label in enumerate(positions):
            if offset % 2 == 1 and offset != n - 1:
                frame.loc[label, "role"] = "test"
    return frame


def assert_disjoint(frame):
    """No contract may appear in both sets. Raises rather than warns."""
    calibration = frame[frame["role"] == "calibration"]
    test = frame[frame["role"] == "test"]
    key = ["expiry", "strike"]
    overlap = calibration.merge(test[key], on=key, how="inner")
    if len(overlap):
        raise ValueError(f"{len(overlap)} contracts are in both sets")
    return True


# --------------------------------------------------------------------------
# The volatility smile
# --------------------------------------------------------------------------

class VolatilitySmile:
    """Implied volatility against log-moneyness, through calibration points.

    PCHIP for the same reason the pricing grid uses it: it passes through every
    point without the overshoot a cubic spline can produce between them, and a
    volatility that dips below zero between two positive quotes would be
    unusable.

    Log-moneyness log(K/S0) rather than the strike, so the curve does not
    change shape when the spot moves.
    """

    def __init__(self, log_moneyness, volatilities):
        from scipy.interpolate import PchipInterpolator

        log_moneyness = np.asarray(log_moneyness, dtype=float)
        volatilities = np.asarray(volatilities, dtype=float)
        finite = np.isfinite(log_moneyness) & np.isfinite(volatilities)
        log_moneyness, volatilities = log_moneyness[finite], volatilities[finite]

        order = np.argsort(log_moneyness)
        self.x = log_moneyness[order]
        self.y = volatilities[order]
        if self.x.size < 2:
            raise ValueError("a smile needs at least two calibration points")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("calibration log-moneyness values must be distinct")

        self._interp = PchipInterpolator(self.x, self.y, extrapolate=False)

    @property
    def n_points(self):
        return int(self.x.size)

    @property
    def range(self):
        return float(self.x[0]), float(self.x[-1])

    def covers(self, log_moneyness):
        """True where a query is inside the calibrated span, not beyond it."""
        x = np.asarray(log_moneyness, dtype=float)
        return (x >= self.x[0]) & (x <= self.x[-1])

    def __call__(self, log_moneyness):
        """Interpolated volatility. Outside the calibrated span this is NaN.

        Deliberately not extrapolated: a volatility invented past the last
        quote would be presented as a prediction while resting on nothing.
        """
        x = np.asarray(log_moneyness, dtype=float)
        return np.asarray(self._interp(x), dtype=float)


def calibrate_smile(calibration_frame, risk_free_rate, dividend_yield=0.0,
                    n_steps=500):
    """Solve American implied volatilities for the calibration contracts.

    Returns a copy with an `implied_vol` column; contracts whose price is not
    attainable within the search bracket are marked NaN and reported rather
    than dropped without trace.
    """
    frame = calibration_frame.copy()
    vols, notes = [], []
    for _, row in frame.iterrows():
        try:
            vols.append(implied_volatility_american_put(
                row["mid"], row["spot"], row["strike"], row["time_to_expiry"],
                risk_free_rate, dividend_yield, n_steps=n_steps))
            notes.append("")
        except (ValueError, ZeroDivisionError) as exc:
            vols.append(np.nan)
            notes.append(str(exc))
    frame["implied_vol"] = vols
    frame["calibration_note"] = notes
    return frame


def smiles_by_expiry(calibration_frame):
    """One VolatilitySmile per expiry, built only from calibration points."""
    smiles = {}
    for expiry, block in calibration_frame.groupby("expiry"):
        usable = block[np.isfinite(block["implied_vol"])]
        if len(usable) < 2:
            continue
        smiles[expiry] = VolatilitySmile(usable["log_moneyness"].to_numpy(),
                                         usable["implied_vol"].to_numpy())
    return smiles


# --------------------------------------------------------------------------
# Predicting the held-out contracts
# --------------------------------------------------------------------------

def predict_heldout(test_frame, smiles, risk_free_rate, dividend_yield=0.0,
                    binomial_steps=1000, lsmc_paths=20_000, lsmc_steps=50,
                    lsmc_degree=2, seed=42):
    """Price every held-out contract from its neighbours' volatility.

    The only market number that reaches the pricer is the interpolated
    volatility, which was fitted without this contract. Its own mid is read
    afterwards, purely to score the prediction.
    """
    import pandas as pd

    rows = []
    for index, (_, row) in enumerate(test_frame.iterrows()):
        smile = smiles.get(row["expiry"])
        record = dict(row)

        if smile is None or not bool(smile.covers(row["log_moneyness"])):
            record.update({"interpolated_vol": np.nan, "crr_price": np.nan,
                           "lsmc_price": np.nan, "lsmc_std_error": np.nan,
                           "prediction_status": "outside the calibrated span"})
            rows.append(record)
            continue

        sigma = float(smile(row["log_moneyness"]))
        if not np.isfinite(sigma) or sigma <= 0:
            record.update({"interpolated_vol": sigma, "crr_price": np.nan,
                           "lsmc_price": np.nan, "lsmc_std_error": np.nan,
                           "prediction_status": "non-positive interpolated vol"})
            rows.append(record)
            continue

        crr = crr_american_put(row["spot"], row["strike"], row["time_to_expiry"],
                               risk_free_rate, sigma, dividend_yield,
                               n_steps=binomial_steps)
        lsmc = price_american_put_lsmc(
            S0=row["spot"], K=row["strike"], T=row["time_to_expiry"],
            r=risk_free_rate, sigma=sigma, q=dividend_yield,
            n_paths=lsmc_paths, n_steps=lsmc_steps, degree=lsmc_degree,
            seed=seed + index, antithetic=True)

        record.update({
            "interpolated_vol": sigma,
            "crr_price": crr,
            "lsmc_price": lsmc.price,
            "lsmc_std_error": lsmc.std_error,
            "prediction_status": "ok",
        })
        rows.append(record)

    frame = pd.DataFrame(rows)
    for model in ("crr", "lsmc"):
        error = frame[f"{model}_price"] - frame["mid"]
        frame[f"{model}_error"] = error
        frame[f"{model}_abs_error"] = error.abs()
        frame[f"{model}_rel_error"] = np.where(
            frame["mid"] >= MIN_PRICE_FOR_PERCENTAGE, error / frame["mid"],
            np.nan)
    return frame


def no_arbitrage_violations(frame, price_column, risk_free_rate):
    """Held-out predictions that break a bound a put price cannot break.

    An American put is worth at least its intrinsic value and never more than
    the strike. These are not model assumptions; a price outside them would be
    a coding error, so they are checked rather than trusted.
    """
    price = frame[price_column].to_numpy(dtype=float)
    intrinsic = np.maximum(frame["strike"].to_numpy(dtype=float)
                           - frame["spot"].to_numpy(dtype=float), 0.0)
    strike = frame["strike"].to_numpy(dtype=float)
    finite = np.isfinite(price)
    below = finite & (price < intrinsic - 1e-8)
    above = finite & (price > strike + 1e-8)
    return {"below_intrinsic": int(below.sum()),
            "above_strike": int(above.sum()),
            "total": int((below | above).sum())}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def error_metrics(actual, predicted, label="", min_price=MIN_PRICE_FOR_PERCENTAGE):
    """Absolute and percentage accuracy over one set of held-out contracts.

    Percentage figures use the median rather than the mean, and only over
    contracts priced above `min_price`: a mean percentage error is dominated by
    whichever contract happened to be cheapest.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    usable = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[usable], predicted[usable]

    if a.size == 0:
        return {"label": label, "n": 0, "mae": np.nan, "rmse": np.nan,
                "bias": np.nan, "median_abs_pct_error": np.nan,
                "max_abs_error": np.nan, "n_for_percentage": 0}

    error = p - a
    meaningful = a >= min_price
    mape = (float(np.median(np.abs(error[meaningful] / a[meaningful]))) * 100.0
            if meaningful.any() else float("nan"))

    return {
        "label": label,
        "n": int(a.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()),
        "median_abs_pct_error": mape,
        "max_abs_error": float(np.abs(error).max()),
        "n_for_percentage": int(meaningful.sum()),
    }


def metrics_table(frame, models=("crr", "lsmc"), group_column="group"):
    """Overall accuracy and accuracy per maturity group, for each model."""
    import pandas as pd

    rows = []
    for model in models:
        rows.append({"model": model, "scope": "overall",
                     **error_metrics(frame["mid"], frame[f"{model}_price"],
                                     "overall")})
        for group, block in frame.groupby(group_column):
            rows.append({"model": model, "scope": str(group),
                         **error_metrics(block["mid"], block[f"{model}_price"],
                                         str(group))})
    return pd.DataFrame(rows).drop(columns=["label"])


def extreme_contracts(frame, model="crr"):
    """The best and worst held-out prediction, for honest reporting."""
    usable = frame[np.isfinite(frame[f"{model}_abs_error"])]
    if usable.empty:
        return None, None
    return (usable.loc[usable[f"{model}_abs_error"].idxmax()],
            usable.loc[usable[f"{model}_abs_error"].idxmin()])
