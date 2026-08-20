"""Interpolating the American-put pricing grid (scope section 15).

A risk scenario lands on some spot such as 743.17, which is not a grid node, so
the option value there has to be interpolated. This is the extra numerical
method the scope asks the extension to add, and it is worth doing carefully:
the whole hedged-portfolio loss distribution is built out of these values.

Three schemes are offered so the choice can be justified rather than assumed:

  linear   safe and monotone, but kinks at the nodes
  cubic    smooth natural cubic spline; can overshoot near the kink at the
           exercise boundary, where the price curve is only C1
  pchip    shape-preserving cubic; smooth AND monotone, so it cannot invent a
           put that gets more valuable as the underlying rises

Outside the grid the interpolant is not trusted at all. A put price is boxed in
by max(K - S, 0) <= P <= K, and far below the exercise boundary an American put
is worth exactly its intrinsic value, so those bounds are applied directly.
"""
import numpy as np

_METHODS = ("linear", "cubic", "pchip")


class PricingGridInterpolator:
    """Callable interpolant over a PricingGrid, with no-arbitrage clamping."""

    def __init__(self, grid, method="pchip"):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        self.grid = grid
        self.method = method
        self._lo = float(grid.spots[0])
        self._hi = float(grid.spots[-1])

        if method == "linear":
            self._interp = lambda s: np.interp(s, grid.spots, grid.prices)
        elif method == "cubic":
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(grid.spots, grid.prices, bc_type="natural")
            self._interp = spline
        else:
            from scipy.interpolate import PchipInterpolator
            self._interp = PchipInterpolator(grid.spots, grid.prices,
                                             extrapolate=True)

    def __call__(self, spots):
        """Interpolated American put value at each spot.

        Values are clamped into [max(K - S, 0), K]. That is not cosmetic: an
        unclamped spline can return a negative price just past the last node,
        which would silently corrupt the loss distribution downstream.
        """
        spots = np.asarray(spots, dtype=float)
        values = np.asarray(self._interp(spots), dtype=float)

        intrinsic = np.maximum(self.grid.strike - spots, 0.0)
        return np.clip(values, intrinsic, self.grid.strike)

    def out_of_range(self, spots):
        """Count how many spots fall outside the grid, so it is never silent."""
        spots = np.asarray(spots, dtype=float)
        below = int(np.sum(spots < self._lo))
        above = int(np.sum(spots > self._hi))
        return {"below": below, "above": above, "total": below + above,
                "fraction": (below + above) / spots.size if spots.size else 0.0,
                "grid_min": self._lo, "grid_max": self._hi}


def assess_interpolation_accuracy(grid, check_spots, reference_prices,
                                  methods=_METHODS, relative_floor=0.01):
    """Compare each interpolation scheme with directly computed prices.

    Scope section 15: verify the interpolation at 10-20 points by pricing them
    directly and comparing.

    Parameters
    ----------
    check_spots : array_like
        Spots strictly inside the grid, chosen away from the nodes so the
        comparison is not trivially exact.
    reference_prices : array_like
        Directly computed prices at those spots.
    relative_floor : float
        Relative errors are reported only where the reference price is at
        least this large. Deep out of the money the true price is a fraction
        of a cent, so a harmless absolute error of 1e-4 shows up as a relative
        error of thousands of percent -- a number that says nothing about
        accuracy and everything about dividing by almost zero.

    Returns
    -------
    list of dicts, one per method, with max/mean absolute error, relative
    error over the meaningful points, and the signed bias.
    """
    check_spots = np.asarray(check_spots, dtype=float)
    reference_prices = np.asarray(reference_prices, dtype=float)
    if check_spots.size != reference_prices.size:
        raise ValueError("check_spots and reference_prices must match in size")

    meaningful = np.abs(reference_prices) >= relative_floor

    results = []
    for method in methods:
        interpolated = PricingGridInterpolator(grid, method)(check_spots)
        errors = interpolated - reference_prices

        if meaningful.any():
            relative = (np.abs(errors[meaningful])
                        / np.abs(reference_prices[meaningful]))
            max_rel, mean_rel = float(relative.max()), float(relative.mean())
        else:
            max_rel = mean_rel = float("nan")

        results.append({
            "method": method,
            "max_absolute_error": float(np.abs(errors).max()),
            "mean_absolute_error": float(np.abs(errors).mean()),
            "max_relative_error": max_rel,
            "mean_relative_error": mean_rel,
            "n_points_for_relative": int(meaningful.sum()),
            "bias": float(errors.mean()),
        })
    return results


def random_check_spots(grid, n_points, rng, margin=0.02):
    """Draw check spots inside the grid, kept clear of the outer edges.

    Points are jittered off the nodes on purpose: an interpolant reproduces its
    own nodes exactly, so testing there would prove nothing.
    """
    lo, hi = grid.spot_range
    span = hi - lo
    return np.sort(rng.uniform(lo + margin * span, hi - margin * span, n_points))
