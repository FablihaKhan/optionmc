"""Tests for the out-of-sample cross-section validation.

The failure this whole module exists to prevent is silent and flattering: a
held-out contract's own price leaking into the volatility used to predict it,
producing an impressively small error that means nothing. That gets a direct
test -- move a held-out quote and check the prediction does not budge -- rather
than a comment claiming it cannot happen.
"""
import numpy as np
import pandas as pd
import pytest

from src.binomial import crr_american_put
from src.cross_section_validation import (VolatilitySmile, apply_quote_filters,
                                          assert_disjoint,
                                          build_contract_frame,
                                          calibrate_smile,
                                          calibration_at_spacing,
                                          error_metrics, metrics_table,
                                          no_arbitrage_violations,
                                          predict_heldout,
                                          select_cross_section_expiries,
                                          smiles_by_expiry,
                                          split_calibration_test,
                                          thin_by_spacing)
from src.market_data import MarketDataError

SPOT = 500.0
RATE = 0.04
DIVIDEND = 0.01
MATURITY = 0.25
EXPIRY = "2026-11-20"
DAYS = 91


def true_sigma(strike, spot=SPOT):
    """A downward-sloping skew: cheaper strikes trade at higher volatility."""
    return 0.20 - 0.45 * np.log(strike / spot)


def make_chain(strikes=None, sigma_fn=true_sigma, half_spread=0.01,
               overrides=None):
    """A put chain quoted at CRR prices along a known smile."""
    if strikes is None:
        strikes = np.arange(455.0, 526.0, 5.0)
    rows = []
    for strike in strikes:
        mid = crr_american_put(SPOT, float(strike), MATURITY, RATE,
                               float(sigma_fn(strike)), DIVIDEND, n_steps=300)
        rows.append({
            "contractSymbol": f"T{strike:g}P",
            "strike": float(strike),
            "bid": mid - half_spread,
            "ask": mid + half_spread,
            "lastPrice": mid,
            "impliedVolatility": float(sigma_fn(strike)),
            "volume": 25.0,
            "openInterest": 300.0,
        })
    frame = pd.DataFrame(rows)
    if overrides:
        for strike, changes in overrides.items():
            mask = frame["strike"] == strike
            for column, value in changes.items():
                frame.loc[mask, column] = value
    return frame


def universe(**kwargs):
    return build_contract_frame(make_chain(**kwargs), SPOT, EXPIRY, DAYS,
                                "long", "2026-08-18", "TEST",
                                min_moneyness=0.90, max_moneyness=1.05)


def pipeline(frame, spacing=None, lsmc_paths=4_000, seed=11):
    kept, report = apply_quote_filters(frame)
    split = split_calibration_test(thin_by_spacing(kept, spacing))
    assert_disjoint(split)
    calibration = calibrate_smile(split[split["role"] == "calibration"],
                                  RATE, DIVIDEND, n_steps=300)
    smiles = smiles_by_expiry(calibration)
    heldout = predict_heldout(split[split["role"] == "test"], smiles, RATE,
                              DIVIDEND, binomial_steps=300,
                              lsmc_paths=lsmc_paths, lsmc_steps=25, seed=seed)
    return split, calibration, smiles, heldout, report


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------

def test_calibration_and_test_sets_do_not_overlap():
    split, _, _, _, _ = pipeline(universe())
    assert_disjoint(split)
    calibration = set(split[split["role"] == "calibration"]["strike"])
    test = set(split[split["role"] == "test"]["strike"])
    assert calibration and test
    assert not (calibration & test)


def test_split_is_deterministic():
    a = split_calibration_test(universe())
    b = split_calibration_test(universe())
    pd.testing.assert_series_equal(a["role"], b["role"])


def test_first_and_last_strike_always_calibrate():
    """Otherwise the outermost held-out strike would be extrapolated."""
    for count in (5, 6, 7, 8):
        strikes = np.arange(470.0, 470.0 + 5.0 * count, 5.0)
        split = split_calibration_test(universe(strikes=strikes))
        ordered = split.sort_values("strike")
        assert ordered["role"].iloc[0] == "calibration"
        assert ordered["role"].iloc[-1] == "calibration"


def test_every_held_out_strike_is_bracketed_by_calibration_strikes():
    split = split_calibration_test(universe())
    calibration = split[split["role"] == "calibration"]["strike"].to_numpy()
    for strike in split[split["role"] == "test"]["strike"]:
        assert calibration.min() < strike < calibration.max()


def test_a_short_strike_list_is_not_forced_into_a_split():
    split = split_calibration_test(universe(strikes=[490.0, 500.0]))
    assert set(split["role"]) == {"calibration"}


# --------------------------------------------------------------------------
# No leakage -- the reason this module exists
# --------------------------------------------------------------------------

def test_a_held_out_quote_does_not_influence_its_own_prediction():
    """Move a held-out contract's market price and its prediction must not move.

    If the held-out mid reached the volatility used to price it, doubling that
    mid would drag the prediction along. The prediction is required to be
    bit-identical; only the recorded error may change.
    """
    base_split, _, _, base_pred, _ = pipeline(universe())
    victim = float(base_split[base_split["role"] == "test"]["strike"].iloc[2])

    original = float(base_pred.loc[base_pred["strike"] == victim,
                                   "crr_price"].iloc[0])
    original_mid = float(base_pred.loc[base_pred["strike"] == victim,
                                       "mid"].iloc[0])

    disturbed = universe(overrides={victim: {"bid": original_mid * 2 - 0.01,
                                             "ask": original_mid * 2 + 0.01}})
    _, _, _, moved_pred, _ = pipeline(disturbed)
    moved = float(moved_pred.loc[moved_pred["strike"] == victim,
                                 "crr_price"].iloc[0])

    assert moved == original
    # The error must move, or the test would pass on a prediction that ignores
    # the market entirely.
    moved_error = float(moved_pred.loc[moved_pred["strike"] == victim,
                                       "crr_error"].iloc[0])
    base_error = float(base_pred.loc[base_pred["strike"] == victim,
                                     "crr_error"].iloc[0])
    assert abs(moved_error - base_error) > 0.5 * original_mid


def test_disturbing_a_calibration_quote_does_change_predictions():
    """The mirror image: calibration quotes are supposed to matter."""
    base_split, _, _, base_pred, _ = pipeline(universe())
    calibration_strikes = base_split[base_split["role"] == "calibration"]["strike"]
    victim = float(calibration_strikes.iloc[len(calibration_strikes) // 2])

    disturbed = universe(overrides={victim: {"bid": 30.0, "ask": 30.02}})
    _, _, _, moved_pred, _ = pipeline(disturbed)

    assert not np.allclose(base_pred["crr_price"].to_numpy(),
                           moved_pred["crr_price"].to_numpy())


def test_assert_disjoint_catches_a_contaminated_split():
    frame = universe()
    split = split_calibration_test(frame)
    duplicate = split[split["role"] == "calibration"].iloc[[0]].copy()
    duplicate["role"] = "test"
    with pytest.raises(ValueError, match="both sets"):
        assert_disjoint(pd.concat([split, duplicate], ignore_index=True))


# --------------------------------------------------------------------------
# Quote filters
# --------------------------------------------------------------------------

def test_filters_remove_crossed_and_empty_quotes_and_say_so():
    frame = universe(overrides={
        470.0: {"bid": 9.0, "ask": 8.0},          # crossed
        480.0: {"bid": 0.0, "ask": 0.0},          # no market
        490.0: {"bid": np.nan, "ask": np.nan},    # missing
    })
    kept, report = apply_quote_filters(frame)
    assert report.n_removed == 3
    assert set(kept["strike"]).isdisjoint({470.0, 480.0, 490.0})
    assert sum(report.reasons.values()) == 3
    assert "crossed" in " ".join(report.reasons)


def test_spread_filter_is_optional_and_counted():
    frame = universe(half_spread=0.01)
    frame.loc[frame["strike"] == 500.0, "ask"] += 5.0
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["spread_percent"] = frame["spread"] / frame["mid"] * 100.0

    loose, loose_report = apply_quote_filters(frame)
    tight, tight_report = apply_quote_filters(frame, max_spread_percent=5.0)
    assert loose_report.n_removed == 0
    assert tight_report.n_removed == 1
    assert len(tight) == len(loose) - 1


def test_filter_report_reads_as_a_sentence():
    frame = universe(overrides={470.0: {"bid": 0.0, "ask": 0.0}})
    _, report = apply_quote_filters(frame)
    assert "kept" in str(report) and "removed" in str(report)


# --------------------------------------------------------------------------
# The smile
# --------------------------------------------------------------------------

def test_smile_passes_through_its_calibration_points():
    x = np.array([-0.10, -0.05, 0.0, 0.04])
    y = np.array([0.24, 0.22, 0.20, 0.19])
    smile = VolatilitySmile(x, y)
    assert smile(x) == pytest.approx(y)


def test_smile_refuses_to_extrapolate():
    smile = VolatilitySmile([-0.1, 0.0, 0.1], [0.24, 0.20, 0.18])
    assert np.isnan(smile(-0.5))
    assert np.isnan(smile(0.5))
    assert not smile.covers(-0.5)
    assert smile.covers(0.0)


def test_smile_stays_positive_between_positive_points():
    x = np.linspace(-0.2, 0.1, 7)
    y = np.array([0.31, 0.27, 0.24, 0.21, 0.20, 0.19, 0.185])
    smile = VolatilitySmile(x, y)
    dense = smile(np.linspace(x[0], x[-1], 400))
    assert np.all(dense > 0)
    # PCHIP is shape preserving: a monotone input may not develop a bump.
    assert np.all(np.diff(dense) <= 1e-12)


def test_smile_needs_at_least_two_points():
    with pytest.raises(ValueError):
        VolatilitySmile([0.0], [0.2])


def test_calibration_recovers_the_smile_the_chain_was_quoted_at():
    split = split_calibration_test(universe())
    calibration = calibrate_smile(split[split["role"] == "calibration"], RATE,
                                  DIVIDEND, n_steps=300)
    expected = true_sigma(calibration["strike"].to_numpy())
    assert calibration["implied_vol"].to_numpy() == pytest.approx(expected,
                                                                  abs=2e-3)


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

def test_every_held_out_contract_gets_a_finite_positive_volatility():
    _, _, _, heldout, _ = pipeline(universe())
    ok = heldout[heldout["prediction_status"] == "ok"]
    assert len(ok) == len(heldout)
    assert np.all(np.isfinite(ok["interpolated_vol"]))
    assert np.all(ok["interpolated_vol"] > 0)


def test_both_pricers_return_finite_prices():
    _, _, _, heldout, _ = pipeline(universe())
    assert np.all(np.isfinite(heldout["crr_price"]))
    assert np.all(np.isfinite(heldout["lsmc_price"]))


def test_predictions_respect_the_american_put_bounds():
    _, _, _, heldout, _ = pipeline(universe())
    for column in ("crr_price", "lsmc_price"):
        assert no_arbitrage_violations(heldout, column, RATE)["total"] == 0


def test_predictions_rise_with_the_strike():
    _, _, _, heldout, _ = pipeline(universe())
    ordered = heldout.sort_values("strike")
    assert ordered["crr_price"].is_monotonic_increasing


def test_lsmc_and_crr_agree_within_the_simulation_noise():
    _, _, _, heldout, _ = pipeline(universe(), lsmc_paths=60_000, seed=3)
    gap = (heldout["lsmc_price"] - heldout["crr_price"]).abs()
    assert np.all(gap <= 4.0 * heldout["lsmc_std_error"] + 0.02)


def test_prediction_is_reproducible_with_a_fixed_seed():
    _, _, _, first, _ = pipeline(universe(), seed=99)
    _, _, _, second, _ = pipeline(universe(), seed=99)
    np.testing.assert_array_equal(first["lsmc_price"].to_numpy(),
                                  second["lsmc_price"].to_numpy())


# --------------------------------------------------------------------------
# Thinning and density
# --------------------------------------------------------------------------

def test_thinning_respects_the_requested_spacing():
    frame = universe(strikes=np.arange(455.0, 526.0, 1.0))
    thinned = thin_by_spacing(frame, 5.0)
    gaps = np.diff(np.sort(thinned["strike"].to_numpy()))
    assert np.all(gaps >= 5.0 - 1e-9)
    assert len(thinned) < len(frame)


def test_thinning_is_a_no_op_without_a_spacing():
    frame = universe()
    assert len(thin_by_spacing(frame, None)) == len(frame)
    assert len(thin_by_spacing(frame, 0)) == len(frame)


def test_calibration_pool_keeps_its_end_strikes():
    """A held-out strike beyond the last calibration point would be dropped."""
    frame = universe(strikes=np.arange(455.0, 526.0, 1.0))
    for spacing in (5.0, 13.0, 40.0):
        pool = calibration_at_spacing(frame, spacing)
        assert pool["strike"].min() == frame["strike"].min()
        assert pool["strike"].max() == frame["strike"].max()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_match_a_hand_computation():
    metrics = error_metrics([1.0, 2.0, 3.0], [1.1, 1.8, 3.3])
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.01 + 0.04 + 0.09) / 3))
    assert metrics["bias"] == pytest.approx((0.1 - 0.2 + 0.3) / 3)
    assert metrics["max_abs_error"] == pytest.approx(0.3)


def test_cheap_contracts_are_excluded_from_percentage_errors():
    """A one-cent error on a three-cent option is not a 33% model failure."""
    metrics = error_metrics([0.03, 5.0, 6.0], [0.04, 5.05, 6.05],
                            min_price=0.10)
    assert metrics["n"] == 3
    assert metrics["n_for_percentage"] == 2
    assert metrics["median_abs_pct_error"] < 2.0


def test_metrics_survive_an_empty_input():
    metrics = error_metrics([], [])
    assert metrics["n"] == 0
    assert np.isnan(metrics["mae"])


def test_metrics_table_reports_overall_and_per_group():
    _, _, _, heldout, _ = pipeline(universe())
    table = metrics_table(heldout)
    assert set(table["model"]) == {"crr", "lsmc"}
    assert "overall" in set(table["scope"])
    assert len(table) == 2 * (1 + heldout["group"].nunique())


# --------------------------------------------------------------------------
# Offline reuse and expiry choice
# --------------------------------------------------------------------------

def test_snapshot_round_trips_through_csv(tmp_path):
    frame = universe()
    path = tmp_path / "snapshot.csv"
    frame.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    _, _, _, from_memory, _ = pipeline(frame)
    _, _, _, from_disk, _ = pipeline(reloaded)
    np.testing.assert_allclose(from_memory["crr_price"].to_numpy(),
                               from_disk["crr_price"].to_numpy())


def test_expiry_selection_picks_one_date_per_maturity_group():
    from datetime import date
    expiries = ["2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18"]
    chosen = select_cross_section_expiries(expiries, date(2026, 8, 19))
    assert [c["expiry"] for c in chosen] == ["2026-09-18", "2026-10-16",
                                             "2026-11-20"]
    assert [c["label"] for c in chosen] == ["short", "medium", "long"]


def test_expiry_selection_never_returns_the_same_date_twice():
    from datetime import date
    chosen = select_cross_section_expiries(["2026-10-02"], date(2026, 8, 19))
    assert len(chosen) == 1


def test_expiry_selection_rejects_an_empty_board():
    from datetime import date
    with pytest.raises(MarketDataError):
        select_cross_section_expiries([], date(2026, 8, 19))


def test_contract_frame_keeps_only_the_moneyness_band():
    frame = build_contract_frame(make_chain(strikes=np.arange(400.0, 601.0, 10.0)),
                                 SPOT, EXPIRY, DAYS, "long", "2026-08-18",
                                 "TEST", min_moneyness=0.90, max_moneyness=1.05)
    assert frame["moneyness"].min() >= 0.90
    assert frame["moneyness"].max() <= 1.05
    assert frame["log_moneyness"].to_numpy() == pytest.approx(
        np.log(frame["strike"].to_numpy() / SPOT))
