"""Shared state that survives moving between pages.

Everything the pages agree on lives in `st.session_state` and nowhere else. No
module-level mutable holds a number: Streamlit reruns the script on every
interaction and shares the process across sessions, so a global would leak one
viewer's parameters into another's and would be almost impossible to reproduce.

The defaults come from `config`, so the dashboard opens on exactly the
parameters the batch pipeline ran with. A viewer who changes nothing sees the
numbers in the report.
"""
from dataclasses import dataclass

import streamlit as st

import config

CACHED = "Cached presentation snapshot"
LIVE = "Live market data"

GBM = "GBM Monte Carlo"
BOOTSTRAP = "Historical bootstrap"

DEFAULTS = {
    "ticker": config.TICKER,
    "expiry": None,             # filled from the snapshot on first load
    "strike": None,
    "n_paths": config.LSMC_N_PATHS,
    "n_steps": config.LSMC_N_STEPS,
    "degree": config.LSMC_DEGREE,
    "confidence_level": 0.99,
    "risk_model": GBM,
    "seed": config.SEED,
    "data_mode": CACHED,        # never fetches on page load
    "explain_simply": False,
    "last_refresh": None,
    "refresh_error": None,
}


def initialise(snapshot=None):
    """Put every key in place once, without overwriting a viewer's choice."""
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    if snapshot is not None:
        # Only fill the blanks: if the viewer has already picked a strike, a
        # rerun must not drag them back to the snapshot's.
        if st.session_state.get("expiry") is None:
            st.session_state["expiry"] = snapshot.expiry
        if st.session_state.get("strike") is None:
            st.session_state["strike"] = snapshot.strike


def get(key):
    return st.session_state.get(key, DEFAULTS.get(key))


def set_value(key, value):
    st.session_state[key] = value


def is_cached_mode():
    return get("data_mode") == CACHED


def explain_simply():
    return bool(get("explain_simply"))


def reset():
    """Back to the defaults, for a viewer who has experimented too far."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


@dataclass(frozen=True)
class PricingParams:
    """The inputs an expensive pricing run needs, frozen so it can be cached.

    Hashable on purpose: `st.cache_data` keys on the arguments, and a frozen
    dataclass of plain numbers is a stable key. Passing `st.session_state`
    around instead would key on an object that changes every rerun and the
    cache would never hit.
    """
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    n_paths: int
    n_steps: int
    degree: int
    seed: int

    @classmethod
    def from_snapshot(cls, snapshot, **overrides):
        values = dict(
            spot=float(snapshot.spot),
            strike=float(get("strike") or snapshot.strike),
            time_to_expiry=float(snapshot.time_to_expiry),
            risk_free_rate=float(snapshot.risk_free_rate),
            dividend_yield=float(snapshot.dividend_yield),
            volatility=float(snapshot.historical_volatility),
            n_paths=int(get("n_paths")),
            n_steps=int(get("n_steps")),
            degree=int(get("degree")),
            seed=int(get("seed")),
        )
        values.update(overrides)
        return cls(**values)
