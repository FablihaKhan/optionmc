"""Cox-Ross-Rubinstein binomial tree -- the American-option benchmark.

An American put has no closed-form Black-Scholes solution, so the extension
needs a different yardstick (scope section 9). The base OptionMC project
validates its European Monte Carlo against Black-Scholes; we keep the same
validation philosophy and validate our LSMC against a CRR tree.

CRR parameters, with dividend yield q:

    dt = T / N
    u  = exp(sigma sqrt(dt)),    d = 1 / u
    p  = (exp((r - q) dt) - d) / (u - d)

The tree is recombining, so level i holds i + 1 nodes and the whole backward
induction is vectorised over a single numpy array.
"""
import numpy as np


def _crr_parameters(T, r, sigma, q, n_steps):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.6f} outside (0,1); "
            "increase n_steps or check sigma/r/q"
        )
    return dt, u, d, p


def _terminal_prices(S0, u, d, n_steps):
    """Prices at maturity: S0 * u^j * d^(n-j) for j = 0..n."""
    j = np.arange(n_steps + 1)
    return S0 * (u ** j) * (d ** (n_steps - j))


def crr_put(S0, K, T, r, sigma, q=0.0, n_steps=1000, american=True):
    """Price a put on a CRR tree.

    Parameters
    ----------
    american : bool
        True  -> early exercise allowed at every node (American).
        False -> exercise only at maturity (European), useful as a
                 cross-check against Black-Scholes.

    Returns
    -------
    float
        Option price at time zero.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt, u, d, p = _crr_parameters(T, r, sigma, q, n_steps)
    discount = np.exp(-r * dt)

    values = np.maximum(K - _terminal_prices(S0, u, d, n_steps), 0.0)

    for i in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(i + 1)
            spot = S0 * (u ** j) * (d ** (i - j))
            np.maximum(values, K - spot, out=values)

    return float(values[0])


def crr_american_put(S0, K, T, r, sigma, q=0.0, n_steps=1000):
    """American put price on a CRR tree (the benchmark for LSMC)."""
    return crr_put(S0, K, T, r, sigma, q, n_steps, american=True)


def crr_bermudan_put(S0, K, T, r, sigma, q=0.0, n_steps=2000,
                     n_exercise_dates=50):
    """Put exercisable only on a coarse grid of dates, priced on a fine tree.

    This is the exact benchmark for an LSMC run using `n_exercise_dates` time
    steps: the LSMC can only ever exercise on its own grid, so comparing it
    with a continuously exercisable American price mixes two different errors
    together. Pricing on a fine tree while restricting exercise to the coarse
    grid separates them:

        LSMC(m)  vs  Bermudan(m)   -> the Monte Carlo error alone
        Bermudan(m) vs American    -> the time-discretisation error alone

    Parameters
    ----------
    n_steps : int
        Tree resolution. Must be a multiple of `n_exercise_dates`.
    n_exercise_dates : int
        Number of equally spaced exercise dates, the last one at maturity.
        Time zero is an exercise opportunity too, exactly as it is in the
        LSMC, whose price is max(continuation, intrinsic).
    """
    if n_exercise_dates < 1:
        raise ValueError("n_exercise_dates must be at least 1")
    if n_steps % n_exercise_dates != 0:
        raise ValueError(
            f"n_steps ({n_steps}) must be a multiple of n_exercise_dates "
            f"({n_exercise_dates}) so the exercise dates land on tree nodes"
        )

    dt, u, d, p = _crr_parameters(T, r, sigma, q, n_steps)
    discount = np.exp(-r * dt)
    stride = n_steps // n_exercise_dates

    values = np.maximum(K - _terminal_prices(S0, u, d, n_steps), 0.0)

    for i in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if i > 0 and i % stride == 0:
            j = np.arange(i + 1)
            spot = S0 * (u ** j) * (d ** (i - j))
            np.maximum(values, K - spot, out=values)

    # At time zero the holder may also exercise immediately.
    return float(max(values[0], K - S0, 0.0))


def crr_european_put(S0, K, T, r, sigma, q=0.0, n_steps=1000):
    """European put price on a CRR tree (should converge to Black-Scholes)."""
    return crr_put(S0, K, T, r, sigma, q, n_steps, american=False)


def early_exercise_premium(S0, K, T, r, sigma, q=0.0, n_steps=1000):
    """American minus European on the same tree -- the value of early exercise."""
    american = crr_american_put(S0, K, T, r, sigma, q, n_steps)
    european = crr_european_put(S0, K, T, r, sigma, q, n_steps)
    return american - european


def implied_volatility_american_put(price, S0, K, T, r, q=0.0, n_steps=500,
                                    lo=1e-4, hi=3.0):
    """Volatility that makes the CRR American put reproduce an observed price.

    The right implied volatility for an American contract: inverting the
    European Black-Scholes formula on an American quote mixes up the two
    payoffs and biases the answer by the early-exercise premium.

    Raises
    ------
    ValueError
        If no volatility in [lo, hi] reproduces `price`.
    """
    from scipy.optimize import brentq

    # A CRR tree only has a valid risk-neutral probability while the up move
    # outruns the drift, i.e. sigma > |r - q| sqrt(dt). Raise the lower bracket
    # above that floor, otherwise the search starts on a degenerate tree.
    dt = T / n_steps
    lo = max(lo, 1.05 * abs(r - q) * np.sqrt(dt))

    def objective(sigma):
        return crr_american_put(S0, K, T, r, sigma, q, n_steps) - price

    if objective(lo) >= 0.0 or objective(hi) <= 0.0:
        raise ValueError(
            f"American put price {price} is not attainable for sigma in "
            f"[{lo}, {hi}]"
        )
    return float(brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10))
