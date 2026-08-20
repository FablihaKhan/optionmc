"""Least-Squares Monte Carlo for American puts (Longstaff & Schwartz, 2001).

This is the main methodological extension. At every exercise date the holder
compares immediate exercise with the value of continuing. The continuation
value is not observable, so it is estimated by regressing realised discounted
future cash flows on a polynomial basis of the current stock price, using the
cross-sectional information in the simulated paths.

Faithful to the paper (and to scope section 23, "do not"):

  * The regression uses ONLY in-the-money paths. The exercise decision is
    irrelevant elsewhere, and restricting the sample sharpens the fit exactly
    where it matters (Longstaff & Schwartz, section 2.2).
  * Y is the REALISED discounted cash flow along each path, never the
    previously estimated continuation value. Discounting back estimated
    conditional expectations biases the price upward (their footnote 9).
  * Nothing here calls a library American-option pricer. The algorithm is
    implemented from the paper.

The regressor is normalised as S/K, which the authors recommend to avoid
scaling problems. For a polynomial basis this leaves the fitted values
mathematically unchanged -- it only improves the conditioning of the
least-squares solve.
"""
from dataclasses import dataclass, field

import numpy as np

from .gbm import risk_neutral_drift, simulate_gbm_paths


@dataclass
class LSMCResult:
    """Everything the experiments and plots need from one LSMC run."""
    price: float
    std_error: float
    european_price: float
    european_std_error: float
    early_exercise_premium: float
    n_paths: int
    n_steps: int
    degree: int
    stopping_step: np.ndarray = field(repr=False)      # per path; n_steps if never early
    exercised_early: np.ndarray = field(repr=False)    # bool per path
    exercise_boundary: np.ndarray = field(repr=False)  # per node, nan where none
    n_itm: np.ndarray = field(repr=False)              # in-the-money count per node
    coefficients: list = field(repr=False)             # regression betas per node

    @property
    def early_exercise_fraction(self):
        """Share of paths exercised strictly before maturity."""
        return float(self.exercised_early.mean())


def lsmc_american_put_from_paths(paths, K, r, dt, degree=2):
    """Core Longstaff-Schwartz recursion on a given path matrix.

    Kept separate from path generation so it can be driven by any set of paths
    -- including the eight-path worked example in section 1 of the paper, which
    the test suite uses to verify this implementation.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps + 1)
        Risk-neutral price paths; column 0 is S0.
    K : float
        Strike.
    r : float
        Risk-free rate (continuously compounded).
    dt : float
        Time between exercise dates, in years.
    degree : int
        Polynomial degree of the continuation-value basis. degree=2 gives
        the scope's first specification C(S) = a0 + a1 S + a2 S^2.

    Returns
    -------
    LSMCResult
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be 2-D (n_paths, n_steps + 1)")
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    if n_steps < 1:
        raise ValueError("paths must contain at least one time step")
    if degree < 1:
        raise ValueError("degree must be at least 1")

    discount_step = np.exp(-r * dt)
    S0 = float(paths[0, 0])

    # cash[i] is the value AT THE CURRENT TIME LEVEL of following the stopping
    # rule identified so far on path i. It starts as the maturity payoff.
    cash = np.maximum(K - paths[:, -1], 0.0)
    stopping_step = np.full(n_paths, n_steps, dtype=int)

    exercise_boundary = np.full(n_nodes, np.nan)
    n_itm = np.zeros(n_nodes, dtype=int)
    coefficients = [None] * n_nodes

    # Work backwards over the early-exercise dates. t = 0 is excluded: the
    # time-zero decision is handled after the loop.
    for t in range(n_steps - 1, 0, -1):
        cash *= discount_step          # now valued at time t
        spot = paths[:, t]
        itm = spot < K                 # in-the-money paths only
        n_itm[t] = int(itm.sum())

        if n_itm[t] < degree + 1:
            # Not enough points to identify the regression; hold everywhere.
            continue

        x = spot[itm] / K              # normalised regressor
        y = cash[itm] / K              # realised discounted cash flow
        design = np.vander(x, degree + 1, increasing=True)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        coefficients[t] = beta

        continuation = (design @ beta) * K
        exercise = K - spot[itm]
        take = exercise > continuation

        if take.any():
            idx = np.flatnonzero(itm)[take]
            cash[idx] = exercise[take]         # exercising kills later cash flows
            stopping_step[idx] = t
            exercise_boundary[t] = float(spot[idx].max())

    path_values = cash * discount_step         # discount from t=1 back to t=0
    continuation_value = float(path_values.mean())
    std_error = float(path_values.std(ddof=1) / np.sqrt(n_paths))

    # At time zero the holder also compares with immediate exercise.
    intrinsic = max(K - S0, 0.0)
    price = max(continuation_value, intrinsic)

    european_payoffs = np.maximum(K - paths[:, -1], 0.0) * np.exp(-r * dt * n_steps)
    european_price = float(european_payoffs.mean())
    european_se = float(european_payoffs.std(ddof=1) / np.sqrt(n_paths))

    return LSMCResult(
        price=price,
        std_error=std_error,
        european_price=european_price,
        european_std_error=european_se,
        early_exercise_premium=price - european_price,
        n_paths=n_paths,
        n_steps=n_steps,
        degree=degree,
        stopping_step=stopping_step,
        exercised_early=stopping_step < n_steps,
        exercise_boundary=exercise_boundary,
        n_itm=n_itm,
        coefficients=coefficients,
    )


def price_american_put_lsmc(S0, K, T, r, sigma, q=0.0, n_paths=10_000,
                            n_steps=50, degree=2, seed=None, antithetic=True,
                            rng=None):
    """Simulate risk-neutral paths and price an American put by LSMC.

    The drift is r - q -- the RISK-NEUTRAL measure (scope section 7). Never
    pass a historical mu here.

    Parameters
    ----------
    seed : int or None
        Seeds a fresh generator, for reproducibility.
    rng : numpy.random.Generator or None
        Supply an existing generator instead of `seed`, e.g. to price a whole
        pricing grid with common random numbers.

    Returns
    -------
    LSMCResult
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(
        S0=S0,
        drift=risk_neutral_drift(r, q),
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=rng,
        antithetic=antithetic,
    )
    return lsmc_american_put_from_paths(paths, K=K, r=r, dt=T / n_steps,
                                        degree=degree)
