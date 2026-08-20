#!/usr/bin/env python
"""PHASE 10: generate the twelve required figures (scope section 19).

Reads the tables and cached distributions the earlier phases wrote, computes
only what is not already on disk, and writes every figure as PNG and PDF at
300 dpi into results/figures/ alongside the CSV it was drawn from.

    python experiments/make_figures.py

Run the earlier experiments first; this script says which one is missing rather
than inventing data to plot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import plots
from src.binomial import crr_american_put
from src.european_mc import reproduce_baseline
from src.gbm import risk_neutral_drift, simulate_gbm_paths
from src.interpolation import PricingGridInterpolator
from src.lsmc import price_american_put_lsmc
from src.market_data import MarketSnapshot
from src.pricing_grid import load_grid

FIG = config.FIGURES_DIR
TAB = config.TABLES_DIR
SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
GRID_JSON = config.DATA_DIR / "pricing_grid.json"
LOSSES_NPZ = config.DATA_DIR / "portfolio_losses.npz"

REQUIRED = {
    "experiment1_convergence.csv": "experiments/convergence.py",
    "experiment2_discretization.csv": "experiments/discretization_test.py",
    "experiment3_regression.csv": "experiments/regression_test.py",
    "pricing_grid.csv": "experiments/interpolation_test.py",
    "interpolation_check_points.csv": "experiments/interpolation_test.py",
    "risk_var_cvar.csv": "experiments/portfolio_risk.py",
    "risk_var_cvar_percent.csv": "experiments/portfolio_risk.py",
}


def check_inputs():
    missing = []
    if not SNAPSHOT_JSON.exists():
        missing.append(("data/market_snapshot.json",
                        "experiments/fetch_market_data.py"))
    for name, script in REQUIRED.items():
        if not (TAB / name).exists():
            missing.append((f"results/tables/{name}", script))
    if not GRID_JSON.exists():
        missing.append(("data/pricing_grid.json",
                        "experiments/interpolation_test.py"))
    if not LOSSES_NPZ.exists():
        missing.append(("data/portfolio_losses.npz",
                        "experiments/portfolio_risk.py"))
    return missing


def main():
    missing = check_inputs()
    if missing:
        print("Cannot draw the figures: inputs are missing.\n")
        for path, script in missing:
            print(f"  {path:<45} run {script}")
        print("\nOr run main.py, which runs every phase in order.")
        return 1

    plots.apply_style()
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)
    horizon_years = config.RISK_HORIZON_DAYS / config.TRADING_DAYS_PER_YEAR
    T_remaining = snap.time_to_expiry - horizon_years
    written = []

    def record(name, fig):
        png, _ = plots.save(fig, FIG, name)
        written.append(png.name)
        print(f"  [{len(written):2d}] {png.name}")

    print("=" * 78)
    print("PHASE 10  Figures")
    print("=" * 78)
    print(f"  SPY American put, K={snap.strike}, exp {snap.expiry}, "
          f"as of {snap.as_of}\n")

    # --- 1. sample simulated SPY paths ------------------------------------
    rng = np.random.default_rng(config.SEED)
    paths = simulate_gbm_paths(
        S0=snap.spot, drift=risk_neutral_drift(snap.risk_free_rate,
                                               snap.dividend_yield),
        sigma=snap.historical_volatility, T=snap.time_to_expiry,
        n_steps=config.LSMC_N_STEPS, n_paths=2_000, rng=rng, antithetic=True)
    times = np.linspace(0.0, snap.time_to_expiry, config.LSMC_N_STEPS + 1)
    fig, _ = plots.plot_sample_paths(
        times, paths, strike=snap.strike,
        title="Risk-neutral SPY price paths used by the LSMC")
    record("01_sample_spy_paths", fig)

    # --- 2. European Monte Carlo convergence (the base project) -----------
    base_sizes = np.logspace(2, 5.3, 12).astype(int)
    base_rows = [reproduce_baseline(S0=100.0, K=100.0, T=1.0, r=0.05,
                                    sigma=0.2, iterations=int(n),
                                    seed=config.SEED + i)
                 for i, n in enumerate(base_sizes)]
    base = pd.DataFrame(base_rows)      # already carries an "iterations" column
    base.to_csv(TAB / "figure02_european_convergence.csv", index=False)
    fig, _ = plots.plot_convergence(
        base_sizes, base["standard_call"].to_numpy(),
        benchmark=float(base["bs_call"].iloc[0]),
        benchmark_label="Black-Scholes",
        title="European call: base OptionMC convergence (reproduced)",
        series_label="standard Monte Carlo")
    record("02_european_mc_convergence", fig)

    # --- 3. LSMC American put convergence ---------------------------------
    ex1 = pd.read_csv(TAB / "experiment1_convergence.csv")
    fig, _ = plots.plot_convergence(
        ex1["n_paths"].to_numpy(), ex1["mean_price"].to_numpy(),
        benchmark=float(ex1["benchmark"].iloc[0]),
        benchmark_label="binomial (CRR)",
        lower=(ex1["mean_price"] - 2 * ex1["std_price"]).to_numpy(),
        upper=(ex1["mean_price"] + 2 * ex1["std_price"]).to_numpy(),
        title=(f"American put: LSMC convergence "
               f"({config.N_REPLICATIONS} replications per point)"),
        series_label="LSMC mean price")
    record("03_lsmc_convergence", fig)

    # --- 4. LSMC against the binomial benchmark ---------------------------
    grid_frame = pd.read_csv(TAB / "pricing_grid.csv")
    fig, _ = plots.plot_lsmc_vs_binomial(
        grid_frame["spot"].to_numpy(),
        grid_frame["american_put_price"].to_numpy(),
        grid_frame["binomial_price"].to_numpy(), strike=snap.strike,
        title="LSMC against the binomial benchmark across spot")
    record("04_lsmc_vs_binomial", fig)

    # --- 5. paths against runtime -----------------------------------------
    fig, _ = plots.plot_runtime(
        ex1["n_paths"].to_numpy(), ex1["mean_runtime_sec"].to_numpy(),
        title="LSMC runtime against number of paths")
    record("05_paths_vs_runtime", fig)

    # --- 6. paths against pricing error -----------------------------------
    from src.replication import fit_convergence_order
    fit = fit_convergence_order(ex1["n_paths"].to_numpy(),
                                ex1["rmse"].to_numpy())
    fig, _ = plots.plot_error_convergence(
        ex1["n_paths"].to_numpy(), ex1["rmse"].to_numpy(),
        fitted_order=fit["order"],
        title="LSMC pricing error against number of paths")
    record("06_paths_vs_error", fig)

    # --- 7. regression degree comparison ----------------------------------
    ex3 = pd.read_csv(TAB / "experiment3_regression.csv")
    degrees = sorted(ex3["degree"].unique())
    path_counts = sorted(ex3["n_paths"].unique())
    series = [ex3[ex3["degree"] == d].sort_values("n_paths")["mean_price"].to_numpy()
              for d in degrees]
    labels = [ex3[ex3["degree"] == d]["basis"].iloc[0] for d in degrees]
    fig, _ = plots.plot_ordered_series(
        path_counts, series, labels,
        benchmark=float(ex3["benchmark"].iloc[0]),
        benchmark_label="Bermudan benchmark",
        title="Regression basis: does a higher degree help?",
        xlabel="number of paths")
    record("07_regression_degree", fig)

    # --- 8. early exercise boundary ---------------------------------------
    boundary_run = price_american_put_lsmc(
        S0=snap.spot, K=snap.strike, T=snap.time_to_expiry,
        r=snap.risk_free_rate, sigma=snap.historical_volatility,
        q=snap.dividend_yield, n_paths=200_000,
        n_steps=config.LSMC_N_STEPS, degree=config.GRID_DEGREE,
        seed=config.SEED, antithetic=True)
    step_times = np.linspace(0.0, snap.time_to_expiry,
                             config.LSMC_N_STEPS + 1)
    remaining = snap.time_to_expiry - step_times
    pd.DataFrame({
        "step": np.arange(config.LSMC_N_STEPS + 1),
        "time_remaining": remaining,
        "exercise_boundary": boundary_run.exercise_boundary,
        "in_the_money_paths": boundary_run.n_itm,
    }).to_csv(TAB / "figure08_exercise_boundary.csv", index=False)
    fig, _ = plots.plot_exercise_boundary(
        remaining, boundary_run.exercise_boundary, snap.strike,
        title=("Early exercise boundary estimated by LSMC  "
               f"({boundary_run.early_exercise_fraction:.1%} of paths exercise early)"))
    record("08_exercise_boundary", fig)

    # --- 9. pricing grid and interpolation --------------------------------
    grid = load_grid(GRID_JSON)
    interpolate = PricingGridInterpolator(grid, config.INTERPOLATION_METHOD)
    dense = np.linspace(*grid.spot_range, 1200)
    checks = pd.read_csv(TAB / "interpolation_check_points.csv")
    fig, _ = plots.plot_pricing_grid(
        grid.spots, grid.prices, dense, interpolate(dense), snap.strike,
        check_spots=checks["spot"].to_numpy(),
        check_prices=checks["binomial"].to_numpy(),
        title=(f"American put pricing grid at the {config.RISK_HORIZON_DAYS}-day "
               f"horizon (T = {T_remaining:.4f} yr)"))
    record("09_pricing_grid_interpolation", fig)

    # --- 10. portfolio loss distributions ---------------------------------
    losses = np.load(LOSSES_NPZ)
    risk_dollars = pd.read_csv(TAB / "risk_var_cvar.csv")
    fig, _ = plots.plot_loss_histogram(
        losses["losses_unhedged"], losses["losses_protected"],
        "SPY only", "SPY + put",
        var_a=float(risk_dollars.iloc[0]["var_99"]),
        var_b=float(risk_dollars.iloc[1]["var_99"]),
        title=(f"Loss distribution over {config.RISK_HORIZON_DAYS} trading days "
               f"({config.N_RISK_SCENARIOS:,} scenarios)"))
    record("10_loss_histogram", fig)

    # --- 11 and 12. VaR and CVaR comparison -------------------------------
    risk_percent = pd.read_csv(TAB / "risk_var_cvar_percent.csv")
    levels = config.CONFIDENCE_LEVELS

    for measure, label, name in (("var", "VaR", "11_var_comparison"),
                                 ("cvar", "CVaR", "12_cvar_comparison")):
        columns = [f"{measure}_{level:.0%}".replace("%", "") for level in levels]
        fig, _ = plots.plot_risk_comparison(
            levels,
            [float(risk_percent.iloc[0][c]) * 100 for c in columns],
            [float(risk_percent.iloc[1][c]) * 100 for c in columns],
            "SPY only", "SPY + put", measure=label, as_percent=True,
            title=(f"{label} as a percentage of portfolio value  "
                   f"({config.RISK_HORIZON_DAYS}-day horizon)"))
        record(name, fig)

    print(f"\n  {len(written)} figures written to {FIG}")
    print("  each as PNG and PDF at 300 dpi, with its data in results/tables/")
    if len(written) != 12:
        print(f"\n  WARNING: expected 12 figures, wrote {len(written)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
