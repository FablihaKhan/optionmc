"""Tests for the deterministic stress engine.

An accounting slip here would move every stress number in the same direction
and look entirely plausible, so the table is checked against its own
definitions and against a case where the hedge is worthless -- the test suite
must not only pass for hedges that happen to work.
"""
import numpy as np
import pytest

from src.binomial import crr_american_put
from src.stress_testing import (DEFAULT_SHOCKS, consistency_report,
                                describe_protection, shocked_spots,
                                stress_table)

SPOT = 500.0
STRIKE = 490.0
RATE = 0.04
DIVIDEND = 0.01
SIGMA = 0.22
SHARES = 100
MULTIPLIER = 100

# The contract is ten trading days shorter at the horizon than it is today.
# Pricing both ends at the same maturity would remove time decay, and a
# protective put that never decays would look free in a flat market.
HORIZON_YEARS = 10 / 252
T_REMAINING = 0.15
T_TODAY = T_REMAINING + HORIZON_YEARS


def put_at(spots, maturity=T_REMAINING, strike=STRIKE):
    return np.array([crr_american_put(float(s), strike, maturity, RATE,
                                      SIGMA, DIVIDEND, n_steps=250)
                     for s in np.atleast_1d(spots)])


def put_today(strike=STRIKE):
    return float(put_at([SPOT], T_TODAY, strike)[0])


def worthless_put(spots):
    return np.zeros(np.shape(spots))


@pytest.fixture
def table():
    return stress_table(SPOT, put_price_now=put_today(),
                        put_value_at=put_at, shares=SHARES, contracts=1,
                        multiplier=MULTIPLIER, strike=STRIKE)


# --------------------------------------------------------------------------
# The shocks
# --------------------------------------------------------------------------

def test_shocked_spots_apply_the_percentage_move():
    spots = shocked_spots(100.0, [0.0, -0.10, -0.30])
    np.testing.assert_allclose(spots, [100.0, 90.0, 70.0])


def test_a_total_wipeout_is_rejected():
    with pytest.raises(ValueError):
        shocked_spots(100.0, [-1.0])
    with pytest.raises(ValueError):
        shocked_spots(100.0, [-1.5])


def test_the_default_shocks_are_the_ones_the_scope_names():
    assert DEFAULT_SHOCKS == (0.0, -0.05, -0.10, -0.20, -0.30)


# --------------------------------------------------------------------------
# The table agrees with its own definitions
# --------------------------------------------------------------------------

def test_every_internal_relation_holds(table):
    for description, passed in consistency_report(
            table, SHARES, 1, MULTIPLIER).items():
        assert passed, description


def test_starting_values_are_defined_correctly(table):
    assert table["stock_only_initial"].iloc[0] == pytest.approx(SHARES * SPOT)
    assert table["protected_initial"].iloc[0] == pytest.approx(
        SHARES * SPOT + MULTIPLIER * put_today())
    # The protected portfolio owns something extra, so it must start richer.
    assert (table["protected_initial"] > table["stock_only_initial"]).all()


def test_each_loss_is_measured_from_its_own_start(table):
    """Measuring both from the unhedged start would count the premium twice."""
    np.testing.assert_allclose(
        table["protected_loss"],
        table["protected_initial"] - table["protected_value"])
    np.testing.assert_allclose(
        table["stock_only_loss"],
        table["stock_only_initial"] - table["stock_only_value"])


def test_an_unshocked_market_leaves_the_stock_exactly_flat(table):
    row = table[np.isclose(table["shock"], 0.0)].iloc[0]
    assert row["stock_only_loss"] == pytest.approx(0.0)
    # The protected portfolio still loses: the put has ten days less life.
    assert row["protected_loss"] > 0


def test_the_stock_loss_is_proportional_to_the_shock(table):
    for _, row in table.iterrows():
        assert row["stock_only_loss_percent"] == pytest.approx(
            -row["shock"] * 100.0)


def test_the_put_gains_value_as_the_market_falls(table):
    ordered = table.sort_values("shock")
    assert ordered["put_value_per_share"].is_monotonic_decreasing


def test_the_hedge_benefit_grows_with_the_crash(table):
    falls = table[table["shock"] < 0].sort_values("shock", ascending=False)
    assert falls["hedge_benefit_dollars"].is_monotonic_increasing


def test_percentages_are_relative_to_each_portfolios_own_value(table):
    np.testing.assert_allclose(
        table["protected_loss_percent"],
        table["protected_loss"] / table["protected_initial"] * 100.0)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_a_put_below_intrinsic_value_is_rejected():
    with pytest.raises(ValueError, match="intrinsic"):
        stress_table(SPOT, 10.0, lambda s: np.zeros(np.shape(s)),
                     strike=STRIKE)


def test_a_put_worth_more_than_its_strike_is_rejected():
    with pytest.raises(ValueError, match="more than its strike"):
        stress_table(SPOT, 10.0,
                     lambda s: np.full(np.shape(s), STRIKE * 2.0),
                     strike=STRIKE)


def test_a_negative_put_value_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        stress_table(SPOT, 10.0, lambda s: np.full(np.shape(s), -1.0))


def test_a_valuer_returning_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="one value per"):
        stress_table(SPOT, 10.0, lambda s: np.array([1.0]))


# --------------------------------------------------------------------------
# The narrative is read off the numbers, not assumed
# --------------------------------------------------------------------------

def test_the_description_matches_the_table(table):
    story = describe_protection(table)
    assert story["helps_anywhere"]
    assert story["benefit_grows_with_the_shock"]
    assert story["largest_benefit_shock"] == pytest.approx(-0.30)
    assert story["worst_unhedged_loss_percent"] == pytest.approx(30.0)
    assert (story["worst_protected_loss_percent"]
            < story["worst_unhedged_loss_percent"])
    # Protection is not free: with no move at all the hedge is a pure cost.
    assert story["cost_in_a_flat_market"] < 0


def test_a_worthless_hedge_is_reported_as_never_helping():
    """The suite must not only pass for hedges that work."""
    table = stress_table(SPOT, put_price_now=5.0, put_value_at=worthless_put,
                         shares=SHARES, contracts=1, multiplier=MULTIPLIER)
    story = describe_protection(table)
    assert not story["helps_anywhere"]
    assert np.isnan(story["first_shock_that_helps"])
    assert (table["protected_loss"] > table["stock_only_loss"]).all()


def test_a_deeper_hedge_caps_the_loss_lower():
    """A higher strike must protect more, or the engine has the sign wrong."""
    losses = {}
    for strike in (460.0, 500.0):
        valuer = (lambda k: (lambda s: put_at(s, T_REMAINING, k)))(strike)
        table = stress_table(SPOT, put_today(strike), valuer,
                             shares=SHARES, contracts=1, multiplier=MULTIPLIER,
                             strike=strike)
        losses[strike] = float(
            table[np.isclose(table["shock"], -0.30)]["protected_loss_percent"].iloc[0])
    assert losses[500.0] < losses[460.0]
