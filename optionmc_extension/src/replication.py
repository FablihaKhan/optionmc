"""Replicated LSMC runs and convergence-order fitting.

A single Monte Carlo run tells you very little: the error it happens to show is
one draw from a distribution. The scope's experiments ask questions -- does a
higher polynomial degree really help? does the price stabilise? -- that cannot
be answered from one seed, because the answer would change with the seed.

So every configuration here is run many times with independent seeds, and the
experiments report the distribution of the error rather than one realisation.
"""
import time

import numpy as np

from .lsmc import price_american_put_lsmc


def independent_seeds(base_seed, n_reps):
    """Generate n_reps well-separated seeds from one base seed.

    SeedSequence spreads the entropy properly, so the runs are independent
    while the whole experiment still reproduces from `base_seed` alone.
    """
    return [int(s) for s in np.random.SeedSequence(base_seed).generate_state(n_reps)]


def replicate_lsmc(n_reps, base_seed, **pricer_kwargs):
    """Run the LSMC pricer n_reps times with independent seeds.

    Parameters
    ----------
    n_reps : int
        Number of independent replications.
    base_seed : int
        Seed for the seed generator, so the whole experiment is reproducible.
    **pricer_kwargs
        Passed straight to `price_american_put_lsmc` (S0, K, T, r, sigma, q,
        n_paths, n_steps, degree, antithetic).

    Returns
    -------
    dict of ndarrays: prices, reported_std_errors, runtimes,
    early_exercise_fractions.
    """
    if n_reps < 1:
        raise ValueError("n_reps must be at least 1")

    prices = np.empty(n_reps)
    reported_se = np.empty(n_reps)
    runtimes = np.empty(n_reps)
    early = np.empty(n_reps)

    for i, seed in enumerate(independent_seeds(base_seed, n_reps)):
        start = time.perf_counter()
        result = price_american_put_lsmc(seed=seed, **pricer_kwargs)
        runtimes[i] = time.perf_counter() - start
        prices[i] = result.price
        reported_se[i] = result.std_error
        early[i] = result.early_exercise_fraction

    return {
        "prices": prices,
        "reported_std_errors": reported_se,
        "runtimes": runtimes,
        "early_exercise_fractions": early,
    }


def summarise(prices, benchmark, runtimes=None, reported_std_errors=None):
    """Turn replicated prices into the statistics the scope asks for.

    Bias, standard deviation and RMSE are kept separate on purpose: RMSE mixes
    the two, and for LSMC the interesting question is usually whether the bias
    (from a suboptimal exercise rule) or the noise dominates.
    """
    prices = np.asarray(prices, dtype=float)
    errors = prices - benchmark
    n = prices.size

    summary = {
        "n_replications": n,
        "mean_price": float(prices.mean()),
        "std_price": float(prices.std(ddof=1)) if n > 1 else float("nan"),
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
        "benchmark": float(benchmark),
        "bias": float(errors.mean()),
        "mean_absolute_error": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "relative_rmse": float(np.sqrt(np.mean(errors ** 2)) / abs(benchmark)),
    }
    # Standard error of the mean price across replications: how precisely we
    # know the average, as opposed to how much one run wobbles.
    summary["std_error_of_mean"] = (
        summary["std_price"] / np.sqrt(n) if n > 1 else float("nan"))

    if runtimes is not None:
        runtimes = np.asarray(runtimes, dtype=float)
        summary["mean_runtime_sec"] = float(runtimes.mean())
        summary["total_runtime_sec"] = float(runtimes.sum())
    if reported_std_errors is not None:
        summary["mean_reported_std_error"] = float(
            np.asarray(reported_std_errors, dtype=float).mean())
    return summary


def fit_convergence_order(sizes, errors):
    """Fit log(error) = a + p log(size) and return the exponent p.

    Monte Carlo theory predicts p = -1/2: the error falls as 1/sqrt(N), so
    halving it costs four times the work. This is the quantitative version of
    the convergence claim the base OptionMC paper makes for European options,
    applied here to the American LSMC price.

    Returns
    -------
    dict with order, intercept and r_squared.
    """
    sizes = np.asarray(sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if sizes.size != errors.size:
        raise ValueError("sizes and errors must have the same length")
    if sizes.size < 2:
        raise ValueError("need at least two points to fit an order")
    if np.any(errors <= 0) or np.any(sizes <= 0):
        raise ValueError("sizes and errors must be strictly positive")

    x = np.log(sizes)
    y = np.log(errors)
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "order": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }
