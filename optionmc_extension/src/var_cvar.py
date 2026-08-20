"""Value-at-Risk and Conditional Value-at-Risk (scope section 17).

Losses are positive numbers here: L = V_0 - V_h, so the right tail is the bad
tail.

    VaR_beta   the beta-quantile of the loss distribution -- the loss that is
               not exceeded with probability beta
    CVaR_beta  the average of the losses that do exceed it

CVaR is computed two independent ways, and they must agree:

  1. Empirically, as mean(L | L >= VaR) -- the definition the scope gives.
  2. By minimising the Rockafellar-Uryasev function

         F_beta(alpha) = alpha + (1 - beta)^-1 E[(L - alpha)^+]

     over alpha. Their Theorem 1 proves this minimum IS the beta-CVaR, and
     that the minimiser is the beta-VaR. Nothing about the loss distribution
     is assumed -- no normality, which matters here because a protective put
     deliberately makes the distribution asymmetric.

Computing it both ways turns the risk reference into a working check rather
than a citation: agreement means the empirical estimator is behaving, and the
minimiser recovering the VaR is the theorem reproducing itself on our data.
"""
import numpy as np


def _validate(losses, level):
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty 1-D array")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level}")
    return losses


def value_at_risk(losses, level=0.95):
    """The beta-quantile of the losses."""
    losses = _validate(losses, level)
    return float(np.quantile(losses, level))


def conditional_value_at_risk(losses, level=0.95):
    """Mean of the losses at or beyond the VaR -- the expected shortfall."""
    losses = _validate(losses, level)
    var = np.quantile(losses, level)
    tail = losses[losses >= var]
    if tail.size == 0:                      # every loss below the quantile
        return float(var)
    return float(tail.mean())


def rockafellar_uryasev_objective(losses, alpha, level=0.95):
    """F_beta(alpha) = alpha + (1 - beta)^-1 * mean((L - alpha)^+)."""
    losses = _validate(losses, level)
    excess = np.maximum(losses - alpha, 0.0)
    return float(alpha + excess.mean() / (1.0 - level))


def cvar_by_minimisation(losses, level=0.95):
    """CVaR via Rockafellar & Uryasev Theorem 1, by minimising F_beta.

    Returns
    -------
    dict with cvar (the minimum), alpha_star (the minimiser, which the theorem
    identifies as the VaR) and the empirical VaR for comparison.
    """
    from scipy.optimize import minimize_scalar

    losses = _validate(losses, level)
    lo, hi = float(losses.min()), float(losses.max())
    if lo == hi:                            # degenerate: a single loss value
        return {"cvar": lo, "alpha_star": lo, "empirical_var": lo}

    scale = hi - lo
    result = minimize_scalar(
        lambda a: rockafellar_uryasev_objective(losses, a, level),
        bounds=(lo, hi), method="bounded",
        options={"xatol": 1e-9 * scale})

    return {
        "cvar": float(result.fun),
        "alpha_star": float(result.x),
        "empirical_var": float(np.quantile(losses, level)),
    }


def risk_measures(losses, levels=(0.95, 0.99), initial_value=None):
    """VaR and CVaR at each level, in dollars and optionally as percentages."""
    losses = np.asarray(losses, dtype=float)
    out = {}
    for level in levels:
        var = value_at_risk(losses, level)
        cvar = conditional_value_at_risk(losses, level)
        key = f"{level:.0%}".replace("%", "")
        out[f"var_{key}"] = var
        out[f"cvar_{key}"] = cvar
        if initial_value:
            out[f"var_{key}_pct"] = var / initial_value
            out[f"cvar_{key}_pct"] = cvar / initial_value
    return out


def bootstrap_risk_measures(losses, levels=(0.95, 0.99), n_bootstrap=500,
                            rng=None, seed=None):
    """Bootstrap standard errors and 95% intervals for VaR and CVaR.

    A tail statistic estimated from a finite sample carries real uncertainty,
    and it is concentrated exactly where the data is thinnest: 99% CVaR is an
    average over the worst 1% of scenarios, so at 50,000 scenarios it rests on
    500 numbers. Reporting a risk reduction without this would overstate how
    precisely it is known.
    """
    losses = np.asarray(losses, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    n = losses.size

    draws = {f"{stat}_{level:.0%}".replace("%", ""): np.empty(n_bootstrap)
             for level in levels for stat in ("var", "cvar")}

    for b in range(n_bootstrap):
        sample = losses[rng.integers(0, n, n)]
        for level in levels:
            key = f"{level:.0%}".replace("%", "")
            var = np.quantile(sample, level)
            draws[f"var_{key}"][b] = var
            tail = sample[sample >= var]
            draws[f"cvar_{key}"][b] = tail.mean() if tail.size else var

    summary = {}
    for name, values in draws.items():
        summary[f"{name}_std_error"] = float(values.std(ddof=1))
        summary[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        summary[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return summary


def risk_reduction(unhedged, protected):
    """Percentage reduction: (unhedged - protected) / unhedged * 100.

    Negative means the hedge made this measure worse. The sign is preserved
    rather than clipped: whether the put helps is a result, not an assumption
    (scope section 23).
    """
    if unhedged == 0:
        return float("nan")
    return (unhedged - protected) / unhedged * 100.0


def risk_table(portfolios, levels=(0.95, 0.99), use_percentage=False):
    """Final risk comparison table (scope section 18).

    Parameters
    ----------
    portfolios : sequence of Portfolio
        The first one is treated as the unhedged baseline for the reduction
        column.
    use_percentage : bool
        Measure losses as a fraction of each portfolio's own initial value.
        This is the fair comparison, because a protected portfolio starts out
        worth more -- it owns the put.

    Returns
    -------
    pandas.DataFrame with one row per portfolio and a reduction row.
    """
    import pandas as pd

    rows = []
    for p in portfolios:
        losses = p.percent_losses if use_percentage else p.losses
        row = {"portfolio": p.name, "initial_value": p.initial_value}
        row.update(risk_measures(losses, levels))
        rows.append(row)

    frame = pd.DataFrame(rows)

    baseline = frame.iloc[0]
    reduction = {"portfolio": "risk reduction %", "initial_value": np.nan}
    for level in levels:
        key = f"{level:.0%}".replace("%", "")
        for stat in ("var", "cvar"):
            column = f"{stat}_{key}"
            reduction[column] = risk_reduction(baseline[column],
                                               frame.iloc[-1][column])
    return pd.concat([frame, pd.DataFrame([reduction])], ignore_index=True)
