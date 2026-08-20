"""European Monte Carlo -- reproducing the base OptionMC results.

Scope PHASE 1 requires reproducing the original European Monte Carlo results
before any extension work. This module calls the ORIGINAL OptionMC classes
rather than reimplementing them, so the reproduction is genuine: if the base
package changed, this would change with it.

It also provides a path-based European put, priced from the very same simulated
paths the LSMC uses. That gives an apples-to-apples American-vs-European
comparison for the sanity check "American put price >= European put price"
(scope section 10), free of any cross-simulation noise.
"""
import time

import numpy as np

from optionmc.models import OptionPricing   # the original, unmodified package


def reproduce_baseline(S0, K, T, r, sigma, iterations, seed=None):
    """Run the original OptionMC European pricers and report against Black-Scholes.

    Returns a dict with standard MC, antithetic MC, the analytical prices, the
    errors and the runtimes -- i.e. the base paper's own comparison, rerun.
    """
    if seed is not None:
        np.random.seed(seed)   # the base package uses numpy's global RNG

    pricer = OptionPricing(S0=S0, E=K, T=T, rf=r, sigma=sigma,
                           iterations=iterations)

    t0 = time.perf_counter()
    std_call = pricer.call_option_simulation()
    std_put = pricer.put_option_simulation()
    std_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    anti_call = pricer.antithetic_call_simulation()
    anti_put = pricer.antithetic_put_simulation()
    anti_time = time.perf_counter() - t0

    bs_call, bs_put = pricer.bs_analytical_price()

    return {
        "S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
        "iterations": iterations,
        "standard_call": std_call, "standard_put": std_put,
        "antithetic_call": anti_call, "antithetic_put": anti_put,
        "bs_call": bs_call, "bs_put": bs_put,
        "standard_call_abs_error": abs(std_call - bs_call),
        "standard_put_abs_error": abs(std_put - bs_put),
        "antithetic_call_abs_error": abs(anti_call - bs_call),
        "antithetic_put_abs_error": abs(anti_put - bs_put),
        "standard_call_rel_error": abs(std_call - bs_call) / bs_call,
        "standard_put_rel_error": abs(std_put - bs_put) / bs_put,
        "antithetic_call_rel_error": abs(anti_call - bs_call) / bs_call,
        "antithetic_put_rel_error": abs(anti_put - bs_put) / bs_put,
        "standard_time": std_time, "antithetic_time": anti_time,
    }


def european_put_from_paths(paths, K, r, T):
    """Price a European put from already-simulated paths.

    Only the terminal column is used -- that is what makes it European. Passing
    the same `paths` array to the LSMC gives the two prices a shared set of
    random draws.

    Returns
    -------
    dict with price and std_error.
    """
    payoffs = np.maximum(K - paths[:, -1], 0.0)
    discounted = np.exp(-r * T) * payoffs
    n = discounted.size
    return {
        "price": float(discounted.mean()),
        "std_error": float(discounted.std(ddof=1) / np.sqrt(n)),
    }
