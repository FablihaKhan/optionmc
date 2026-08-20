"""How numbers are written on screen.

One place, so a dollar amount looks the same on every page and a reader never
has to work out whether 0.55 meant 55% or 0.55%. Every function here is pure
and takes no Streamlit dependency, which is also what makes them testable.

Missing values are rendered as an em dash rather than "nan". A dashboard that
prints nan in a metric card has told the viewer nothing and looks broken; a
dash says "not available" and is honest.
"""
import math

MISSING = "—"


def _missing(value):
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def money(value, decimals=2):
    """$1,234.56 -- the scope's format, with thousands separators."""
    if _missing(value):
        return MISSING
    return f"${value:,.{decimals}f}"


def signed_money(value, decimals=2):
    """+$1,234.56 or -$1,234.56, sign always shown.

    Used where the direction is the point: a hedge benefit that is negative in
    a flat market must not read as a gain.
    """
    if _missing(value):
        return MISSING
    return f"{'-' if value < 0 else '+'}${abs(value):,.{decimals}f}"


def percent(value, decimals=2):
    """A value already in percentage units: 55.12 -> 55.12%."""
    if _missing(value):
        return MISSING
    return f"{value:.{decimals}f}%"


def fraction_as_percent(value, decimals=2):
    """A value on 0-1: 0.5512 -> 55.12%."""
    if _missing(value):
        return MISSING
    return f"{value * 100:.{decimals}f}%"


def signed_percent(value, decimals=2):
    if _missing(value):
        return MISSING
    return f"{'-' if value < 0 else '+'}{abs(value):.{decimals}f}%"


def price(value, decimals=4):
    """An option price. Four decimals: cents matter when the put costs $12."""
    if _missing(value):
        return MISSING
    return f"{value:.{decimals}f}"


def ratio(value, decimals=2, suffix="x"):
    if _missing(value):
        return MISSING
    return f"{value:.{decimals}f}{suffix}"


def count(value):
    if _missing(value):
        return MISSING
    return f"{int(value):,}"


def volatility(value, decimals=2):
    """A volatility given as a decimal fraction: 0.172149 -> 17.21%."""
    if _missing(value):
        return MISSING
    return f"{value * 100:.{decimals}f}%"


def strike_label(value):
    if _missing(value):
        return MISSING
    return f"K={value:g}"


def moneyness(value, decimals=1):
    """A strike as a share of spot: 0.9748 -> 97.5% of spot."""
    if _missing(value):
        return MISSING
    return f"{value * 100:.{decimals}f}% of spot"


def days(value):
    if _missing(value):
        return MISSING
    return f"{int(value)} days"


def snapshot_caption(as_of, ticker, days_to_expiry, expiry=None):
    """The badge line: '18 Aug 2026 - SPY - 73 DTE'."""
    from datetime import datetime

    try:
        stamp = datetime.strptime(str(as_of), "%Y-%m-%d").strftime("%d %b %Y")
    except (ValueError, TypeError):
        stamp = str(as_of)
    parts = [stamp, str(ticker), f"{int(days_to_expiry)} DTE"]
    if expiry:
        parts.append(f"expiry {expiry}")
    return "  •  ".join(parts)
