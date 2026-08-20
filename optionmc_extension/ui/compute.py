"""Cached calls into the numerical engine.

The pages ask for results; this module runs them and remembers the answers.
Nothing here implements mathematics -- every function is a thin wrapper around
`src/`, and its job is caching, shaping the output for display, and keeping the
expensive work in one place where the cost is visible.

Caching matters more than it looks. Streamlit re-executes the whole page script
on every interaction, so without `cache_data` switching from one output tab to
another would re-run the simulation. Each function is keyed on a frozen
`PricingParams`, which is a stable tuple of plain numbers; keying on the
session-state object instead would produce a new key every rerun and the cache
would never hit.
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.binomial import crr_american_put, crr_european_put
from src.gbm import risk_neutral_drift, simulate_gbm_paths
from src.lsmc import price_american_put_lsmc
from src.pricing_grid import build_pricing_grid, moneyness_grid

# Path counts the convergence sweep visits, as multiples of the chosen count.
CONVERGENCE_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0)

# Ceiling on a single simulation started from the dashboard. Someone exploring
# the sliders should not be able to launch a run that outlives the lesson.
MAX_PATHS = 200_000


def clamp_paths(n_paths):
    """Keep an interactive run inside a few seconds."""
    return int(min(max(int(n_paths), 100), MAX_PATHS))


@st.cache_data(show_spinner=False)
def price_once(params, binomial_steps=1000):
    """One LSMC valuation, its CRR benchmark, and everything derived from them.

    Returns a plain dict so `cache_data` can pickle it and so the page never
    has to know the shape of an `LSMCResult`.
    """
    n_paths = clamp_paths(params.n_paths)
    result = price_american_put_lsmc(
        S0=params.spot, K=params.strike, T=params.time_to_expiry,
        r=params.risk_free_rate, sigma=params.volatility,
        q=params.dividend_yield, n_paths=n_paths, n_steps=params.n_steps,
        degree=params.degree, seed=params.seed, antithetic=True)

    american = crr_american_put(params.spot, params.strike,
                                params.time_to_expiry, params.risk_free_rate,
                                params.volatility, params.dividend_yield,
                                n_steps=binomial_steps)
    european = crr_european_put(params.spot, params.strike,
                                params.time_to_expiry, params.risk_free_rate,
                                params.volatility, params.dividend_yield,
                                n_steps=binomial_steps)

    difference = result.price - american
    times = np.linspace(0.0, params.time_to_expiry, params.n_steps + 1)

    return {
        "lsmc_price": float(result.price),
        "lsmc_std_error": float(result.std_error),
        "european_price": float(result.european_price),
        "early_exercise_premium": float(result.early_exercise_premium),
        "early_exercise_fraction": float(result.early_exercise_fraction),
        "binomial_american": float(american),
        "binomial_european": float(european),
        "binomial_premium": float(american - european),
        "difference": float(difference),
        "relative_difference": (abs(difference) / american * 100.0
                                if american else float("nan")),
        "within_two_standard_errors": bool(
            abs(difference) <= 2.0 * result.std_error),
        "intrinsic": float(max(params.strike - params.spot, 0.0)),
        "n_paths_used": n_paths,
        "times": times,
        "time_remaining": params.time_to_expiry - times,
        "exercise_boundary": np.asarray(result.exercise_boundary, dtype=float),
        "in_the_money": np.asarray(result.n_itm, dtype=float),
        "stopping_step": np.asarray(result.stopping_step, dtype=float),
        "exercised_early": np.asarray(result.exercised_early, dtype=bool),
    }


@st.cache_data(show_spinner=False)
def sample_paths(params, n_display=160):
    """A handful of the simulated paths, for the picture.

    Drawn from a separate generator so displaying them cannot disturb the
    valuation's own random numbers -- the price on screen must not change
    because someone opened a chart.
    """
    n_display = int(min(max(n_display, 2), 400))
    if n_display % 2:
        n_display += 1
    rng = np.random.default_rng(params.seed + 991)
    paths = simulate_gbm_paths(
        S0=params.spot,
        drift=risk_neutral_drift(params.risk_free_rate, params.dividend_yield),
        sigma=params.volatility, T=params.time_to_expiry,
        n_steps=params.n_steps, n_paths=n_display, rng=rng, antithetic=True)
    times = np.linspace(0.0, params.time_to_expiry, params.n_steps + 1)
    return times, paths


@st.cache_data(show_spinner=False)
def price_across_spots(params, n_points=17, span=0.30, binomial_steps=800):
    """LSMC and CRR across a range of spot values.

    The LSMC side reuses one set of unit paths rescaled to every spot -- common
    random numbers, the same device the pricing grid uses -- so the curve is
    smooth and the comparison against the lattice is not swamped by a different
    random draw at each point.
    """
    spots = moneyness_grid(params.spot, 1.0 - span, 1.0 + span, int(n_points))
    grid = build_pricing_grid(
        spots=spots, K=params.strike, T_remaining=params.time_to_expiry,
        r=params.risk_free_rate, sigma=params.volatility,
        q=params.dividend_yield, n_paths=clamp_paths(params.n_paths),
        n_steps=params.n_steps, degree=params.degree, seed=params.seed,
        antithetic=True)
    binomial = np.array([
        crr_american_put(float(s), params.strike, params.time_to_expiry,
                         params.risk_free_rate, params.volatility,
                         params.dividend_yield, n_steps=binomial_steps)
        for s in spots])
    return pd.DataFrame({
        "spot": spots,
        "moneyness": spots / params.strike,
        "lsmc": grid.prices,
        "lsmc_std_error": grid.std_errors,
        "binomial": binomial,
        "difference": grid.prices - binomial,
        "intrinsic": np.maximum(params.strike - spots, 0.0),
    })


@st.cache_data(show_spinner=False)
def convergence_sweep(params, factors=CONVERGENCE_FACTORS, binomial_steps=1000):
    """Price and error against the number of paths, at the chosen settings.

    One replication per point, which is enough to show the trend on screen but
    not enough to measure the convergence order -- that needs the thirty-seed
    study on the numerical methods page, and the caption says so.
    """
    benchmark = crr_american_put(params.spot, params.strike,
                                 params.time_to_expiry, params.risk_free_rate,
                                 params.volatility, params.dividend_yield,
                                 n_steps=binomial_steps)
    rows = []
    for index, factor in enumerate(factors):
        n_paths = clamp_paths(params.n_paths * factor)
        result = price_american_put_lsmc(
            S0=params.spot, K=params.strike, T=params.time_to_expiry,
            r=params.risk_free_rate, sigma=params.volatility,
            q=params.dividend_yield, n_paths=n_paths, n_steps=params.n_steps,
            degree=params.degree, seed=params.seed + index, antithetic=True)
        rows.append({
            "n_paths": n_paths,
            "price": float(result.price),
            "std_error": float(result.std_error),
            "lower": float(result.price - 2 * result.std_error),
            "upper": float(result.price + 2 * result.std_error),
            "benchmark": float(benchmark),
            "absolute_error": abs(float(result.price) - benchmark),
        })
    return pd.DataFrame(rows).drop_duplicates(subset="n_paths")


def exercise_boundary_frame(result):
    """The boundary as a tidy table, with the nodes that never exercised."""
    return pd.DataFrame({
        "step": np.arange(result["exercise_boundary"].size),
        "time_remaining": result["time_remaining"],
        "exercise_boundary": result["exercise_boundary"],
        "in_the_money_paths": result["in_the_money"],
    })


def summary_frame(params, result):
    """Every headline number of a run, for the raw-results tab."""
    return pd.DataFrame([
        ("Spot", params.spot),
        ("Strike", params.strike),
        ("Time to expiry (years)", params.time_to_expiry),
        ("Risk-free rate", params.risk_free_rate),
        ("Dividend yield", params.dividend_yield),
        ("Volatility", params.volatility),
        ("Paths", result["n_paths_used"]),
        ("Exercise steps", params.n_steps),
        ("Polynomial degree", params.degree),
        ("Seed", params.seed),
        ("LSMC American price", result["lsmc_price"]),
        ("LSMC standard error", result["lsmc_std_error"]),
        ("LSMC European price", result["european_price"]),
        ("LSMC early-exercise premium", result["early_exercise_premium"]),
        ("Paths exercising early", result["early_exercise_fraction"]),
        ("CRR American price", result["binomial_american"]),
        ("CRR European price", result["binomial_european"]),
        ("CRR early-exercise premium", result["binomial_premium"]),
        ("LSMC minus CRR", result["difference"]),
        ("Relative difference (%)", result["relative_difference"]),
        ("Intrinsic value", result["intrinsic"]),
    ], columns=["quantity", "value"])


# --------------------------------------------------------------------------
# The hedge optimizer
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def hedge_candidates_table(risk_model, n_scenarios, horizon_days, seed):
    """Re-run every candidate hedge's risk under the chosen engine.

    The contracts, their calibrated volatilities and their pricing grids all
    come from the cached optimizer run -- rebuilding a grid here would take
    minutes. What is recomputed is the part that depends on the risk model: the
    horizon scenarios, the protected portfolios, and the VaR and CVaR that
    follow.

    Both engines see the SAME scenario set across candidates, so the difference
    between two strikes is the hedge and not the draw.

    Returns None when the optimizer has not been run.
    """
    import config
    from src.hedge_optimizer import HedgeCandidate, candidate_risk
    from src.historical_bootstrap import (bootstrap_horizon_prices,
                                          daily_log_returns)
    from src.market_data import MarketSnapshot, load_spy_history
    from src.portfolio import unhedged_portfolio
    from src.pricing_grid import load_grid
    from src.risk_simulation import horizon_in_years, simulate_horizon_scenarios
    from src.var_cvar import risk_measures

    candidates_csv = config.TABLES_DIR / "hedge_optimizer_candidates.csv"
    snapshot_json = config.DATA_DIR / "market_snapshot.json"
    grids = config.DATA_DIR / "hedge_grids"
    if not candidates_csv.exists() or not snapshot_json.exists():
        return None

    frame = pd.read_csv(candidates_csv)
    snap = MarketSnapshot.from_json(snapshot_json)
    horizon_years = horizon_in_years(horizon_days, config.TRADING_DAYS_PER_YEAR)

    if risk_model == "Historical bootstrap":
        history = load_spy_history(config.SPY_HISTORY_CSV)
        returns = daily_log_returns(history["Close"].dropna().to_numpy())
        spots = bootstrap_horizon_prices(
            S0=snap.spot, log_returns=returns, horizon_days=horizon_days,
            n_scenarios=n_scenarios, seed=seed + 3)
    else:
        spots = simulate_horizon_scenarios(
            S0=snap.spot, real_world_drift=snap.historical_drift,
            sigma=snap.historical_volatility, horizon_days=horizon_days,
            n_scenarios=n_scenarios,
            trading_days_per_year=config.TRADING_DAYS_PER_YEAR,
            seed=seed + 2, antithetic=True)

    baseline = unhedged_portfolio(config.SHARES, snap.spot, spots)
    levels = tuple(config.CONFIDENCE_LEVELS)

    rows, missing = [], []
    for _, entry in frame.iterrows():
        path = grids / f"grid_K{entry['strike']:g}.json"
        if not path.exists():
            missing.append(float(entry["strike"]))
            continue
        candidate = HedgeCandidate(
            target_moneyness=float(entry["target_moneyness"]),
            strike=float(entry["strike"]), moneyness=float(entry["moneyness"]),
            contract_symbol=str(entry["contract_symbol"]),
            expiry=str(entry["expiry"]),
            days_to_expiry=int(entry["days_to_expiry"]),
            bid=float(entry["bid"]), ask=float(entry["ask"]),
            mid=float(entry["mid"]), last_price=float(entry["last_price"]),
            quoted_iv=float(entry["quoted_iv"]),
            price_source=str(entry["price_source"]),
            volume=float(entry["volume"]),
            open_interest=float(entry["open_interest"]),
            as_of=str(entry["as_of"]))
        risk, _ = candidate_risk(
            candidate, load_grid(path), spots, snap.spot, baseline,
            shares=config.SHARES, contracts=1,
            multiplier=config.CONTRACT_MULTIPLIER, levels=levels,
            interpolation_method=config.INTERPOLATION_METHOD)
        row = candidate.to_row()
        row.update(risk)
        row["sigma"] = float(entry["sigma"])
        row["lsmc_price"] = float(entry["lsmc_price"])
        row["binomial_price"] = float(entry["binomial_price"])
        rows.append(row)

    if not rows:
        return None

    dollars = risk_measures(baseline.losses, levels)
    percent = risk_measures(baseline.percent_losses, levels)
    return {
        "candidates": pd.DataFrame(rows),
        "baseline": {
            "initial_value": float(baseline.initial_value),
            **{f"{k}_dollars": v for k, v in dollars.items()},
            **{f"{k}_percent": v * 100.0 for k, v in percent.items()},
        },
        "risk_model": risk_model,
        "n_scenarios": int(n_scenarios),
        "spot": float(snap.spot),
        "expiry": str(snap.expiry),
        "t_remaining": float(snap.time_to_expiry - horizon_years),
        "missing_grids": missing,
    }


def rank_for_objective(frame, level, protection_weight, cost_weight):
    """Score and rank the candidates at the chosen confidence level.

    Cheap by design: the expensive simulation happens once per risk model, and
    changing the objective or the weights only re-scores the table already in
    hand. A viewer can move the slider and watch the answer move without
    waiting for fifty thousand scenarios.
    """
    from src.hedge_optimizer import rank_candidates

    key = f"{level:.0%}".replace("%", "")
    return rank_candidates(
        frame, protection_weight=protection_weight, cost_weight=cost_weight,
        protection_column=f"cvar_{key}_reduction", cost_column="premium_cost")


def objective_winner(frame, winners, objective, level):
    """Which row a named objective selects, at the chosen confidence level."""
    key = f"{level:.0%}".replace("%", "")
    if objective == "Cheapest":
        return frame.loc[frame["premium_cost"].idxmin()]
    if objective == "Strongest protection":
        return frame.loc[frame[f"cvar_{key}_reduction"].idxmax()]
    if objective == "Best efficiency":
        return frame.loc[frame[f"cvar_{key}_saved_per_premium_dollar"].idxmax()]
    return winners["balanced"]


@st.cache_data(show_spinner=False)
def smile_curve(expiry, n_points=400):
    """Rebuild the fitted volatility smile from the saved calibration set.

    Cheap to redo and safer than storing the curve: the shape on screen is
    always the one those calibration quotes produce, so it cannot drift away
    from the table beside it.
    """
    import config
    from src.cross_section_validation import VolatilitySmile

    path = config.TABLES_DIR / "calibration_set.csv"
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    block = frame[(frame["expiry"] == expiry)
                  & np.isfinite(frame["implied_vol"])]
    if len(block) < 2:
        return None

    smile = VolatilitySmile(block["log_moneyness"].to_numpy(),
                            block["implied_vol"].to_numpy())
    dense = np.linspace(*smile.range, int(n_points))
    return {"x": dense, "y": smile(dense),
            "calibration_x": block["log_moneyness"].to_numpy(),
            "calibration_y": block["implied_vol"].to_numpy()}


@st.cache_data(show_spinner=False)
def historical_returns():
    """The observed daily SPY log returns the bootstrap resamples from.

    Returned with their shape statistics, because the whole reason the second
    risk engine exists is that these are not normal.
    """
    import config
    from src.historical_bootstrap import (daily_log_returns,
                                          return_distribution_summary)
    from src.market_data import load_spy_history

    if not config.SPY_HISTORY_CSV.exists():
        return None
    history = load_spy_history(config.SPY_HISTORY_CSV)
    closes = history["Close"].dropna()
    returns = daily_log_returns(closes.to_numpy())
    return {
        "returns": returns,
        "summary": return_distribution_summary(returns, "observed daily"),
        "start": str(closes.index[0].date()),
        "end": str(closes.index[-1].date()),
    }


# --------------------------------------------------------------------------
# Sanity checks, derived from what is on disk right now
# --------------------------------------------------------------------------

def _check(name, passed, detail, source):
    return {"name": name, "passed": bool(passed), "detail": detail,
            "source": source}


@st.cache_data(show_spinner=False)
def sanity_checks():
    """Re-derive every invariant from the saved results.

    Deliberately recomputed rather than read from a stored pass/fail list. A
    saved verdict can outlive the numbers it was about; deriving it here means
    the indicator on screen is always about the tables currently on disk, and
    it turns red the moment they stop satisfying it.

    Each entry carries the file it was checked against, so a green tick can be
    traced rather than trusted.
    """
    import inspect

    import config
    from src.historical_bootstrap import bootstrap_horizon_prices
    from src.market_data import MarketSnapshot

    def table(name):
        path = config.TABLES_DIR / name
        return pd.read_csv(path) if path.exists() else None

    checks = []

    pricing = table("spy_american_put_pricing.csv")
    if pricing is not None and len(pricing):
        row = pricing.iloc[0]
        european = max(float(row["european_bs_price"]),
                       float(row["european_tree_price"]))
        checks.append(_check(
            "American put is worth at least the European one",
            float(row["lsmc_price"]) >= european,
            f"{row['lsmc_price']:.4f} against {european:.4f}; the gap is the "
            f"early-exercise premium",
            "spy_american_put_pricing.csv"))
        checks.append(_check(
            "LSMC and the CRR lattice agree",
            float(row["relative_error_vs_binomial"]) < 0.01,
            f"{row['relative_error_vs_binomial']:.4%} apart, against an LSMC "
            f"standard error of {row['lsmc_std_error']:.4f}",
            "spy_american_put_pricing.csv"))

    grid = table("pricing_grid.csv")
    if grid is not None and len(grid):
        below = int((grid["american_put_price"]
                     < grid["intrinsic_value"] - 1e-9).sum())
        checks.append(_check(
            "American put is never below its intrinsic value",
            below == 0,
            f"{len(grid)} grid nodes checked, {below} below intrinsic",
            "pricing_grid.csv"))
        rising = bool(grid.sort_values("spot")["american_put_price"]
                      .is_monotonic_decreasing)
        checks.append(_check(
            "The put is worth less as the underlying rises",
            rising, f"monotone across all {len(grid)} nodes",
            "pricing_grid.csv"))

    risk = table("risk_var_cvar.csv")
    if risk is not None and len(risk) >= 2:
        breaches = 0
        for _, entry in risk.iloc[:2].iterrows():
            for level in ("95", "99"):
                if entry[f"cvar_{level}"] < entry[f"var_{level}"]:
                    breaches += 1
        checks.append(_check(
            "CVaR is never below VaR",
            breaches == 0,
            f"{2 * len(risk.iloc[:2])} portfolio-level pairs checked, "
            f"{breaches} breaches",
            "risk_var_cvar.csv"))

    snapshot_path = config.DATA_DIR / "market_snapshot.json"
    if snapshot_path.exists():
        snap = MarketSnapshot.from_json(snapshot_path)
        risk_neutral = snap.risk_free_rate - snap.dividend_yield
        separated = (abs(snap.historical_drift - risk_neutral) > 1e-9
                     and abs(snap.historical_drift - snap.risk_free_rate) > 1e-9)
        checks.append(_check(
            "The pricing and risk measures are separated",
            separated,
            f"pricing drift r - q = {risk_neutral:+.6f}, risk drift mu = "
            f"{snap.historical_drift:+.6f}",
            "market_snapshot.json"))

    calibration = table("calibration_set.csv")
    heldout = table("heldout_predictions.csv")
    if calibration is not None and heldout is not None:
        shared = (set(zip(calibration["expiry"], calibration["strike"]))
                  & set(zip(heldout["expiry"], heldout["strike"])))
        checks.append(_check(
            "No contract is in both the calibration and held-out sets",
            not shared,
            f"{len(calibration)} calibration, {len(heldout)} held out, "
            f"{len(shared)} shared",
            "calibration_set.csv, heldout_predictions.csv"))

        ok = heldout[heldout["prediction_status"] == "ok"]
        if len(ok):
            outside = int((heldout["prediction_status"]
                           == "outside the calibrated span").sum())
            checks.append(_check(
                "Every held-out volatility was interpolated, never extrapolated",
                outside == 0,
                f"{len(ok)} predictions inside the fitted span, {outside} past "
                f"its end",
                "heldout_predictions.csv"))

    convergence = table("experiment1_convergence.csv")
    if convergence is not None and len(convergence) >= 3:
        from src.replication import fit_convergence_order

        fit = fit_convergence_order(convergence["n_paths"].to_numpy(),
                                    convergence["rmse"].to_numpy())
        checks.append(_check(
            "The estimator converges at the Monte Carlo rate",
            abs(fit["order"] + 0.5) < 0.2,
            f"fitted order {fit['order']:.3f} against the theoretical -0.5",
            "experiment1_convergence.csv"))

    accuracy = table("interpolation_accuracy.csv")
    if accuracy is not None and len(accuracy):
        chosen = accuracy[(accuracy["method"] == config.INTERPOLATION_METHOD)
                          & (accuracy["stage"] == "a_interpolation_only")]
        if len(chosen):
            worst = float(chosen["max_absolute_error"].iloc[0])
            checks.append(_check(
                "Interpolating the pricing grid costs less than a cent",
                worst < 0.01,
                f"worst {worst:.5f} per share against directly priced points",
                "interpolation_accuracy.csv"))

    frontier = table("protection_cost_frontier.csv")
    if frontier is not None and len(frontier):
        costs = frontier["premium_cost"].to_numpy()
        protection = frontier["cvar_99_reduction"].to_numpy()
        dominated = 0
        for i in np.flatnonzero(frontier["pareto_efficient"].to_numpy()):
            better = ((costs <= costs[i]) & (protection >= protection[i])
                      & ((costs < costs[i]) | (protection > protection[i])))
            dominated += int(np.any(better))
        checks.append(_check(
            "No hedge on the frontier is dominated",
            dominated == 0,
            f"{int(frontier['pareto_efficient'].sum())} efficient candidates, "
            f"{dominated} dominated",
            "protection_cost_frontier.csv"))

    forbidden = {"r", "rate", "risk_free_rate", "drift", "mu", "sigma",
                 "volatility"}
    parameters = set(inspect.signature(bootstrap_horizon_prices).parameters)
    checks.append(_check(
        "The bootstrap engine cannot be handed a risk-free rate",
        not (forbidden & parameters),
        f"its parameters are {', '.join(sorted(parameters))}",
        "src/historical_bootstrap.py"))

    return checks


@st.cache_data(show_spinner=False)
def project_facts():
    """Counts and timestamps derived from the repository, not written down.

    The test count is read by parsing the test files rather than by running
    them: a page load must not start a test suite, and a number typed into the
    source would be stale the moment anyone added a case.

    What it counts is test *functions*. Parametrised ones expand into several
    cases at run time, so the suite reports a larger number, and the caption
    beside it says so.
    """
    import ast
    from datetime import datetime

    import config

    root = config.ROOT_DIR
    tests, files = 0, sorted((root / "tests").glob("test_*.py"))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        tests += sum(1 for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith("test_"))

    def newest(paths):
        stamps = [path.stat().st_mtime for path in paths if path.exists()]
        return (datetime.fromtimestamp(max(stamps)).strftime("%d %b %Y %H:%M")
                if stamps else None)

    tables = sorted(config.TABLES_DIR.glob("*.csv"))
    figures = sorted(config.FIGURES_DIR.glob("*.png"))

    return {
        "test_functions": tests,
        "test_files": len(files),
        "source_modules": len(sorted((root / "src").glob("*.py"))),
        "experiments": len(sorted((root / "experiments").glob("*.py"))),
        "tables": len(tables),
        "figures": len(figures),
        "results_written": newest(tables + figures),
        "ui_modules": len(sorted((root / "ui").glob("*.py")))
                      + len(sorted((root / "ui_pages").glob("*.py"))),
    }
