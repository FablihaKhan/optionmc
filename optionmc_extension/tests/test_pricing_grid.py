"""Pricing grid and interpolation (scope section 15)."""
import numpy as np
import pytest

from src.binomial import crr_american_put
from src.interpolation import (PricingGridInterpolator,
                               assess_interpolation_accuracy,
                               random_check_spots)
from src.pricing_grid import (PricingGrid, build_pricing_grid, moneyness_grid,
                              price_at_spots_directly)

# A short-dated put, so tests stay fast.
CONTRACT = dict(K=100.0, T_remaining=0.25, r=0.04, sigma=0.25, q=0.01)


def _grid(n_points=21, n_paths=20_000, degree=3, seed=42):
    spots = moneyness_grid(100.0, 0.6, 1.4, n_points)
    return build_pricing_grid(spots=spots, n_paths=n_paths, n_steps=25,
                              degree=degree, seed=seed, **CONTRACT)


def test_moneyness_grid_spans_the_requested_range():
    spots = moneyness_grid(768.37, 0.60, 1.40, 33)
    assert spots.size == 33
    assert spots[0] == pytest.approx(0.60 * 768.37)
    assert spots[-1] == pytest.approx(1.40 * 768.37)
    assert np.all(np.diff(spots) > 0)


def test_moneyness_grid_rejects_bad_ranges():
    with pytest.raises(ValueError):
        moneyness_grid(100.0, 0.6, 1.4, 1)
    with pytest.raises(ValueError):
        moneyness_grid(100.0, 1.4, 0.6, 10)
    with pytest.raises(ValueError):
        moneyness_grid(100.0, -0.1, 1.4, 10)


def test_grid_is_monotone_and_above_intrinsic():
    grid = _grid()
    assert grid.is_monotone_decreasing()
    intrinsic = np.maximum(grid.strike - grid.spots, 0.0)
    assert np.all(grid.prices >= intrinsic - 1e-9)
    assert np.all(grid.prices <= grid.strike + 1e-9)


def test_grid_matches_the_binomial_tree():
    grid = _grid(n_paths=50_000)
    tree = np.array([
        crr_american_put(S0=s, K=CONTRACT["K"], T=CONTRACT["T_remaining"],
                         r=CONTRACT["r"], sigma=CONTRACT["sigma"],
                         q=CONTRACT["q"], n_steps=1000)
        for s in grid.spots])
    assert np.abs(grid.prices - tree).max() < 0.05


def test_common_random_numbers_make_the_grid_smooth():
    """With shared shocks the price curve must have no node-to-node jitter.

    A grid priced with independent draws per node wobbles; that wobble would be
    interpolated as if it were signal. Second differences are the sensitive
    test: they explode on jitter long before monotonicity breaks.
    """
    grid = _grid(n_points=41, n_paths=20_000)
    second_difference = np.diff(grid.prices, n=2)
    scale = np.abs(np.diff(grid.prices)).mean()
    assert np.abs(second_difference).max() < scale


def test_grid_is_reproducible_and_seed_sensitive():
    a = _grid(seed=7)
    b = _grid(seed=7)
    c = _grid(seed=8)
    assert np.array_equal(a.prices, b.prices)
    assert not np.array_equal(a.prices, c.prices)


def test_grid_rejects_a_horizon_past_expiry():
    with pytest.raises(ValueError, match="T_remaining must be positive"):
        build_pricing_grid(spots=moneyness_grid(100.0, 0.6, 1.4, 11),
                           K=100.0, T_remaining=-0.01, r=0.04, sigma=0.25)


def test_grid_rejects_unsorted_spots():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_pricing_grid(spots=np.array([100.0, 90.0, 110.0]), K=100.0,
                           T_remaining=0.25, r=0.04, sigma=0.25)
    with pytest.raises(ValueError):
        build_pricing_grid(spots=np.array([100.0]), K=100.0, T_remaining=0.25,
                           r=0.04, sigma=0.25)


def _exact_grid(n_points=21):
    """A grid filled with binomial values, so no Monte Carlo noise is present."""
    spots = moneyness_grid(100.0, 0.6, 1.4, n_points)
    prices = np.array([
        crr_american_put(S0=s, K=CONTRACT["K"], T=CONTRACT["T_remaining"],
                         r=CONTRACT["r"], sigma=CONTRACT["sigma"],
                         q=CONTRACT["q"], n_steps=1000)
        for s in spots])
    return PricingGrid(spots=spots, prices=prices,
                       std_errors=np.zeros_like(prices), strike=100.0,
                       time_to_expiry=0.25, risk_free_rate=0.04,
                       dividend_yield=0.01, volatility=0.25, n_paths=0,
                       n_steps=25, degree=3)


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_interpolant_reproduces_its_own_nodes(method):
    grid = _exact_grid()
    interp = PricingGridInterpolator(grid, method)
    assert np.allclose(interp(grid.spots), grid.prices, atol=1e-9)


def test_interpolation_error_falls_as_the_grid_is_refined():
    """On an exact grid the scheme's error must actually converge."""
    check = np.linspace(65.0, 135.0, 120)
    reference = np.array([
        crr_american_put(S0=s, K=CONTRACT["K"], T=CONTRACT["T_remaining"],
                         r=CONTRACT["r"], sigma=CONTRACT["sigma"],
                         q=CONTRACT["q"], n_steps=1000)
        for s in check])

    errors = []
    for n_points in (11, 21, 41):
        interp = PricingGridInterpolator(_exact_grid(n_points), "pchip")
        errors.append(np.abs(interp(check) - reference).max())
    assert errors[0] > errors[1] > errors[2]


def test_pchip_stays_monotone_where_a_cubic_spline_may_not():
    grid = _exact_grid(21)
    dense = np.linspace(*grid.spot_range, 3000)
    values = PricingGridInterpolator(grid, "pchip")(dense)
    assert np.all(np.diff(values) <= 1e-9)


def test_interpolant_is_clamped_into_the_no_arbitrage_box():
    """Outside the grid a spline can go negative; clamping must prevent that."""
    grid = _exact_grid()
    interp = PricingGridInterpolator(grid, "cubic")
    far = np.array([10.0, 40.0, 200.0, 500.0])
    values = interp(far)
    assert np.all(values >= np.maximum(100.0 - far, 0.0) - 1e-9)
    assert np.all(values <= 100.0 + 1e-9)


def test_out_of_range_is_counted_not_hidden():
    grid = _exact_grid()
    lo, hi = grid.spot_range
    report = PricingGridInterpolator(grid).out_of_range(
        np.array([lo - 1.0, lo + 1.0, hi - 1.0, hi + 1.0, hi + 2.0]))
    assert report["below"] == 1
    assert report["above"] == 2
    assert report["total"] == 3
    assert report["fraction"] == pytest.approx(0.6)


def test_unknown_interpolation_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        PricingGridInterpolator(_exact_grid(), "quadratic")


def test_relative_error_ignores_worthless_options():
    """A 0.001 error on a 0.002 option must not be reported as 50% accuracy."""
    grid = _exact_grid()
    spots = np.array([70.0, 130.0])           # deep in and deep out of the money
    reference = np.array([30.0, 0.002])
    result = assess_interpolation_accuracy(grid, spots, reference,
                                           methods=("pchip",),
                                           relative_floor=0.01)[0]
    assert result["n_points_for_relative"] == 1        # only the ITM point
    assert result["max_relative_error"] < 0.05


def test_assess_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        assess_interpolation_accuracy(_exact_grid(), np.array([90.0, 100.0]),
                                      np.array([5.0]))


def test_random_check_spots_stay_inside_and_off_the_nodes():
    grid = _exact_grid()
    rng = np.random.default_rng(1)
    spots = random_check_spots(grid, 15, rng)
    lo, hi = grid.spot_range
    assert spots.size == 15
    assert np.all(spots > lo) and np.all(spots < hi)
    assert np.all(np.diff(spots) > 0)
    assert not np.any(np.isclose(spots[:, None], grid.spots[None, :]).any(axis=1))


def test_direct_pricing_reuses_the_grid_construction():
    """Priced at a node, the direct route must reproduce that node exactly."""
    grid = _grid(n_points=11, n_paths=20_000, seed=3)
    direct = price_at_spots_directly(grid.spots[3:6], grid, seed=3)
    assert np.allclose(direct, grid.prices[3:6], atol=1e-12)
