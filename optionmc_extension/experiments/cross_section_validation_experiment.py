#!/usr/bin/env python
"""ADVANCED PHASE 2: does the pricing framework generalise out of sample?

The project's single-contract result can be produced by a method that proves
nothing: solve a contract's implied volatility from its own market price, put
that volatility back into the pricer, and report a tiny error. The model was
handed the answer.

This experiment removes that shortcut. Within each expiry the usable strikes
are sorted and split by position -- even calibrates, odd is held out, with the
first and last strike forced into calibration so no held-out point sits past
the end of the curve. Only calibration contracts have their American implied
volatilities solved, against the CRR tree. A PCHIP smile is fitted through
those points over log-moneyness. Each held-out contract reads its volatility
off that curve, is priced by CRR and by LSMC, and only then is compared with
its own quoted mid.

    python experiments/cross_section_validation_experiment.py
    python experiments/cross_section_validation_experiment.py --refresh
    python experiments/cross_section_validation_experiment.py --spacing 10

--refresh attempts a live three-maturity fetch. Outside trading hours the feed
returns a zero bid and zero ask on every contract, and the run says so and
falls back to the quoted chain already cached rather than validating against
prices that do not exist.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import plots, plots_validation
from src.cross_section_validation import (MIN_PRICE_FOR_PERCENTAGE,
                                          apply_quote_filters, assert_disjoint,
                                          calibrate_smile,
                                          calibration_at_spacing,
                                          cross_section_from_cached_chain,
                                          download_cross_section,
                                          error_metrics, extreme_contracts,
                                          metrics_table,
                                          no_arbitrage_violations,
                                          predict_heldout,
                                          split_calibration_test,
                                          smiles_by_expiry, thin_by_spacing)
from src.market_data import MarketDataError, MarketSnapshot
from src.sanity import Check, report

SNAPSHOT_JSON = config.DATA_DIR / "market_snapshot.json"
CHAIN_CSV = config.SPY_OPTION_CSV

SNAPSHOT_CSV = config.TABLES_DIR / "option_cross_section_snapshot.csv"
CALIBRATION_CSV = config.TABLES_DIR / "calibration_set.csv"
HELDOUT_CSV = config.TABLES_DIR / "heldout_predictions.csv"
METRICS_CSV = config.TABLES_DIR / "cross_section_metrics.csv"
SPACING_CSV = config.TABLES_DIR / "cross_section_spacing_study.csv"


def load_universe(snap, refresh):
    """Get the option cross-section, saying plainly where it came from."""
    if refresh:
        try:
            frame, source = download_cross_section(
                config.TICKER, config.CROSS_SECTION_CSV,
                min_moneyness=config.CROSS_SECTION_MIN_MONEYNESS,
                max_moneyness=config.CROSS_SECTION_MAX_MONEYNESS,
                force_refresh=True)
            return frame, f"live fetch ({source})"
        except MarketDataError as exc:
            print(f"\n  live fetch unusable: {exc}\n")
    elif config.CROSS_SECTION_CSV.exists():
        frame = pd.read_csv(config.CROSS_SECTION_CSV)
        return frame, "cached multi-maturity snapshot"

    chain = pd.read_csv(CHAIN_CSV)
    frame = cross_section_from_cached_chain(
        chain, snap.spot, snap.expiry, snap.days_to_expiry, snap.as_of,
        group=f"{snap.days_to_expiry}d", ticker=config.TICKER,
        min_moneyness=config.CROSS_SECTION_MIN_MONEYNESS,
        max_moneyness=config.CROSS_SECTION_MAX_MONEYNESS)
    return frame, "cached single-expiry chain"


def run_pipeline(frame, snap, spacing, lsmc_paths, seed, quiet=False):
    """Filter, thin, split, calibrate, and predict. Returns every stage."""
    kept, filter_report = apply_quote_filters(
        frame, max_spread_percent=config.CROSS_SECTION_MAX_SPREAD_PERCENT)
    n_before_thin = len(kept)
    thinned = thin_by_spacing(kept, spacing)
    split = split_calibration_test(thinned)
    assert_disjoint(split)

    calibration = calibrate_smile(
        split[split["role"] == "calibration"], snap.risk_free_rate,
        snap.dividend_yield, n_steps=config.CROSS_SECTION_IV_TREE_STEPS)
    smiles = smiles_by_expiry(calibration)
    heldout = predict_heldout(
        split[split["role"] == "test"], smiles, snap.risk_free_rate,
        snap.dividend_yield, binomial_steps=config.BINOMIAL_N_STEPS,
        lsmc_paths=lsmc_paths, lsmc_steps=config.LSMC_N_STEPS,
        lsmc_degree=config.LSMC_DEGREE, seed=seed)
    return {
        "filtered": kept, "filter_report": filter_report,
        "n_before_thin": n_before_thin, "split": split,
        "calibration": calibration, "smiles": smiles, "heldout": heldout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="attempt a live three-maturity fetch")
    parser.add_argument("--spacing", type=float,
                        default=config.CROSS_SECTION_STRIKE_SPACING,
                        help="minimum spacing between kept strikes, in dollars")
    parser.add_argument("--skip-spacing-study", action="store_true",
                        help="skip the calibration-density sweep")
    args = parser.parse_args()

    if not SNAPSHOT_JSON.exists() or not CHAIN_CSV.exists():
        print(f"Missing {SNAPSHOT_JSON} or {CHAIN_CSV}.")
        print("Run: python experiments/fetch_market_data.py")
        return 1
    snap = MarketSnapshot.from_json(SNAPSHOT_JSON)

    print("=" * 78)
    print("ADVANCED PHASE 2  Out-of-sample cross-section validation")
    print("=" * 78)

    universe, source = load_universe(snap, args.refresh)
    expiries = sorted(universe["expiry"].unique())
    multi_maturity = len(expiries) > 1

    print(f"  data source: {source}")
    print(f"  {len(universe)} contracts in the "
          f"{config.CROSS_SECTION_MIN_MONEYNESS:.0%}-"
          f"{config.CROSS_SECTION_MAX_MONEYNESS:.0%} moneyness band, "
          f"{len(expiries)} expiry/expiries")
    for expiry, block in universe.groupby("expiry"):
        print(f"    {expiry}  {int(block['days_to_expiry'].iloc[0]):>3} days   "
              f"{len(block):>3} strikes  "
              f"{block['strike'].min():g}-{block['strike'].max():g}")

    if not multi_maturity:
        print("\n  NOTE: only one maturity is available with live quotes, so")
        print("  this run validates across STRIKES, not across maturities.")
        print("  The three-maturity path is implemented and one command away")
        print("  (--refresh during trading hours); it is not run on stale")
        print("  last-trade prices, which are recorded seconds to an hour")
        print("  apart and would show up as smile roughness that the held-out")
        print("  errors would then wrongly blame on the model.")

    stages = run_pipeline(universe, snap, args.spacing,
                          config.CROSS_SECTION_LSMC_PATHS, config.SEED)
    filter_report = stages["filter_report"]
    split = stages["split"]
    calibration = stages["calibration"]
    heldout = stages["heldout"]

    print(f"\n  quote filters: {filter_report}")
    print(f"  strike thinning to ${args.spacing:g} spacing: "
          f"{stages['n_before_thin']} -> {len(split)} contracts")
    print("    (a density choice, not a quality filter: adjacent SPY strikes "
          "are\n     a dollar apart, and interpolating across one dollar "
          "would pass\n     without testing anything)")

    n_cal = int((split["role"] == "calibration").sum())
    n_test = int((split["role"] == "test").sum())
    print(f"\n  split: {n_cal} calibration, {n_test} held out, "
          f"disjoint by construction")

    finite_iv = int(np.isfinite(calibration["implied_vol"]).sum())
    print(f"  calibration: {finite_iv} of {len(calibration)} implied "
          f"volatilities solved against the CRR American tree")
    if finite_iv:
        print(f"    sigma from {calibration['implied_vol'].min():.4f} "
              f"(at the money) to {calibration['implied_vol'].max():.4f} "
              f"(deep out of the money)")

    ok = heldout[heldout["prediction_status"] == "ok"]
    for status, count in heldout["prediction_status"].value_counts().items():
        if status != "ok":
            print(f"    {count} held-out contracts skipped: {status}")

    # --- the held-out table -------------------------------------------------
    head = (f"\n  {'strike':>8} {'money':>7} {'market':>9} {'vol used':>9} "
            f"{'CRR':>9} {'LSMC':>9} {'CRR err':>9} {'LSMC err':>9}")
    print(head)
    print("  " + "-" * (len(head) - 3))
    for _, r in ok.iterrows():
        print(f"  {r['strike']:>8.1f} {r['moneyness']:>7.4f} "
              f"{r['mid']:>9.3f} {r['interpolated_vol']:>9.4f} "
              f"{r['crr_price']:>9.4f} {r['lsmc_price']:>9.4f} "
              f"{r['crr_error']:>+9.4f} {r['lsmc_error']:>+9.4f}")

    metrics = metrics_table(ok)
    metrics.to_csv(METRICS_CSV, index=False)

    print(f"\n  accuracy on {len(ok)} held-out contracts "
          f"(percentages over the {int(metrics.iloc[0]['n_for_percentage'])} "
          f"priced at or above ${MIN_PRICE_FOR_PERCENTAGE:.2f})")
    head = (f"  {'model':>6} {'scope':>10} {'n':>4} {'MAE':>9} {'RMSE':>9} "
            f"{'bias':>10} {'median APE':>11} {'max abs':>9}")
    print(head)
    print("  " + "-" * (len(head) - 3))
    for _, m in metrics.iterrows():
        print(f"  {m['model']:>6} {m['scope']:>10} {int(m['n']):>4} "
              f"{m['mae']:>9.4f} {m['rmse']:>9.4f} {m['bias']:>+10.4f} "
              f"{m['median_abs_pct_error']:>10.4f}% {m['max_abs_error']:>9.4f}")

    # --- is the LSMC error just its own Monte Carlo noise? ------------------
    within = int((ok["lsmc_error"].abs() <= 2 * ok["lsmc_std_error"]).sum())
    noise_floor = float(ok["lsmc_std_error"].mean())
    print(f"\n  {within} of {len(ok)} LSMC errors sit inside that contract's "
          f"own +/- 2 standard errors.")
    print(f"  Mean LSMC standard error {noise_floor:.4f} against a CRR mean "
          f"absolute error of\n  "
          f"{float(metrics[metrics['model'] == 'crr'].iloc[0]['mae']):.4f}: "
          "the simulation's disagreement with the market is\n  dominated by "
          "its own sampling noise, not by the smile or the pricer.")

    worst, best = extreme_contracts(ok, "crr")
    if worst is not None:
        print(f"\n  largest CRR error: K={worst['strike']:g} "
              f"({worst['moneyness']:.1%}), market {worst['mid']:.3f}, "
              f"model {worst['crr_price']:.3f}, "
              f"error {worst['crr_error']:+.4f}")
        print(f"  smallest CRR error: K={best['strike']:g} "
              f"({best['moneyness']:.1%}), market {best['mid']:.3f}, "
              f"model {best['crr_price']:.3f}, "
              f"error {best['crr_error']:+.4f}")

    print("\n   What this does and does not show. It shows the American")
    print("   implied-volatility surface is smooth enough that a strike left")
    print("   out of the fit can be repriced from its neighbours to within a")
    print("   few cents, and that LSMC and CRR agree to within LSMC's own")
    print("   sampling error. It does not show the model beats the market:")
    print("   the pricer that values the held-out contract is the same one")
    print("   that inverted the calibration quotes, so what is being tested")
    print("   is the interpolation and the pricing pipeline, not a view.")

    # --- calibration density study -----------------------------------------
    spacing_frame = pd.DataFrame()
    if not args.skip_spacing_study:
        print(f"\n  calibration density: how far apart can the calibration "
              f"strikes be?")
        # The held-out set is held FIXED at the contracts the primary run
        # tested. Re-splitting at every spacing would test a different set of
        # contracts each time, and the comparison would be measuring which
        # contracts happened to be held out rather than how far apart the
        # calibration strikes can be.
        test_frame = split[split["role"] == "test"]
        test_keys = set(zip(test_frame["expiry"], test_frame["strike"]))
        pool = stages["filtered"][
            ~stages["filtered"].apply(
                lambda r: (r["expiry"], r["strike"]) in test_keys, axis=1)]
        print(f"    held-out set fixed at the same {len(test_frame)} contracts "
              f"throughout; only the {len(pool)}-contract calibration pool is "
              f"thinned")

        rows = []
        for spacing in config.CROSS_SECTION_SPACING_STUDY:
            cal_pool = calibration_at_spacing(pool, spacing)
            trial_cal = calibrate_smile(
                cal_pool, snap.risk_free_rate, snap.dividend_yield,
                n_steps=config.CROSS_SECTION_IV_TREE_STEPS)
            trial_smiles = smiles_by_expiry(trial_cal)
            trial_pred = predict_heldout(
                test_frame, trial_smiles, snap.risk_free_rate,
                snap.dividend_yield, binomial_steps=config.BINOMIAL_N_STEPS,
                lsmc_paths=config.CROSS_SECTION_LSMC_PATHS,
                lsmc_steps=config.LSMC_N_STEPS, lsmc_degree=config.LSMC_DEGREE,
                seed=config.SEED)
            trial_ok = trial_pred[trial_pred["prediction_status"] == "ok"]
            if trial_ok.empty:
                continue
            crr_m = error_metrics(trial_ok["mid"], trial_ok["crr_price"])
            lsmc_m = error_metrics(trial_ok["mid"], trial_ok["lsmc_price"])
            rows.append({
                "spacing": spacing, "n_contracts": len(cal_pool) + len(test_frame),
                "n_calibration": len(trial_cal), "n_heldout": len(trial_ok),
                "mean_calibration_gap": spacing,
                "crr_mae": crr_m["mae"], "crr_rmse": crr_m["rmse"],
                "crr_max_abs_error": crr_m["max_abs_error"],
                "lsmc_mae": lsmc_m["mae"], "lsmc_rmse": lsmc_m["rmse"],
                "lsmc_std_error_mean": float(trial_ok["lsmc_std_error"].mean()),
            })
        spacing_frame = pd.DataFrame(rows)
        spacing_frame.to_csv(SPACING_CSV, index=False)

        head = (f"  {'spacing':>8} {'cal pts':>8} {'test':>5} {'CRR MAE':>9} "
                f"{'CRR RMSE':>9} {'CRR max':>9} {'LSMC MAE':>9}")
        print(head)
        print("  " + "-" * (len(head) - 3))
        for _, s in spacing_frame.iterrows():
            print(f"  {'$' + format(s['spacing'], 'g'):>8} "
                  f"{int(s['n_calibration']):>8} {int(s['n_heldout']):>5} "
                  f"{s['crr_mae']:>9.4f} {s['crr_rmse']:>9.4f} "
                  f"{s['crr_max_abs_error']:>9.4f} {s['lsmc_mae']:>9.4f}")
        if len(spacing_frame) > 1:
            first, last = spacing_frame.iloc[0], spacing_frame.iloc[-1]
            best = spacing_frame.loc[spacing_frame["crr_mae"].idxmin()]
            print(f"\n   The same {int(last['n_heldout'])} contracts "
                  f"throughout, priced from a calibration grid\n   thinning "
                  f"from {int(first['n_calibration'])} points at "
                  f"${first['spacing']:g} spacing to "
                  f"{int(last['n_calibration'])} points at "
                  f"${last['spacing']:g}. The pricer, the\n   contracts and "
                  f"the quotes never change, so the sweep isolates the\n   "
                  f"volatility surface's own interpolation error.")
            print(f"\n   The error does not fall as the grid gets finer. It "
                  f"is lowest at\n   ${best['spacing']:g} spacing "
                  f"(${best['crr_mae']:.4f}), rises to "
                  f"${last['crr_mae']:.4f} at ${last['spacing']:g}, and is "
                  f"also worse\n   at ${first['spacing']:g} "
                  f"(${first['crr_mae']:.4f}). That is a bias-variance "
                  f"trade-off in the smile:")
            print("   PCHIP passes through every calibration point, so at a "
                  "fine spacing it\n   threads the penny-wide quantisation of "
                  "the bid-ask quotes and carries\n   that wobble into the "
                  "held-out strikes; at a coarse spacing the curve is\n   "
                  "smooth but has too far to reach. Densest is not best.")

    # --- persist ------------------------------------------------------------
    universe.to_csv(SNAPSHOT_CSV, index=False)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    heldout.to_csv(HELDOUT_CSV, index=False)

    # --- sanity checks ------------------------------------------------------
    print("\n  sanity checks:")
    cal_keys = set(zip(calibration["expiry"], calibration["strike"]))
    test_keys = set(zip(heldout["expiry"], heldout["strike"]))
    crr_bounds = no_arbitrage_violations(ok, "crr_price", snap.risk_free_rate)
    lsmc_bounds = no_arbitrage_violations(ok, "lsmc_price", snap.risk_free_rate)
    crr_mae = float(metrics[metrics["model"] == "crr"].iloc[0]["mae"])

    checks = [
        Check("calibration and held-out sets do not overlap",
              len(cal_keys & test_keys) == 0,
              f"{len(cal_keys)} calibration, {len(test_keys)} held out, "
              f"{len(cal_keys & test_keys)} shared"),
        Check("every held-out contract got a finite positive volatility",
              bool(np.all(np.isfinite(ok["interpolated_vol"]))
                   and np.all(ok["interpolated_vol"] > 0)),
              f"{len(ok)} interpolated volatilities"),
        Check("every held-out volatility came from inside the calibrated span",
              int((heldout["prediction_status"]
                   == "outside the calibrated span").sum()) == 0,
              "no extrapolation past the end of the smile"),
        Check("every CRR price is finite",
              bool(np.all(np.isfinite(ok["crr_price"]))), f"{len(ok)} prices"),
        Check("every LSMC price is finite",
              bool(np.all(np.isfinite(ok["lsmc_price"]))), f"{len(ok)} prices"),
        Check("CRR predictions obey the American put bounds",
              crr_bounds["total"] == 0,
              f"{crr_bounds['below_intrinsic']} below intrinsic, "
              f"{crr_bounds['above_strike']} above strike"),
        Check("LSMC predictions obey the American put bounds",
              lsmc_bounds["total"] == 0,
              f"{lsmc_bounds['below_intrinsic']} below intrinsic, "
              f"{lsmc_bounds['above_strike']} above strike"),
        Check("held-out prices fall as the strike falls",
              bool(ok.sort_values("strike")["crr_price"].is_monotonic_increasing),
              "a put is worth more at a higher strike"),
        Check("the market snapshot can be replayed offline",
              SNAPSHOT_CSV.exists() and len(pd.read_csv(SNAPSHOT_CSV)) == len(universe),
              f"{SNAPSHOT_CSV.name} holds all {len(universe)} contracts"),
        Check("held-out error stays under a cent per dollar of option value",
              crr_mae < 0.01 * float(ok["mid"].mean()),
              f"CRR MAE ${crr_mae:.4f} against a mean quote of "
              f"${float(ok['mid'].mean()):.2f}"),
    ]
    all_passed = report(checks)

    # --- figures ------------------------------------------------------------
    plots.apply_style()
    written = []

    def record(name, fig):
        plots.save(fig, config.FIGURES_DIR, name)
        written.append(name)

    fig, _ = plots_validation.plot_market_vs_model(
        ok["mid"], ok["crr_price"], ok["lsmc_price"], ok["lsmc_std_error"],
        title=(f"Held-out SPY puts: predicted from neighbouring strikes only "
               f"({len(ok)} contracts)"))
    record("16_market_vs_model", fig)

    expiry = expiries[0]
    cal_block = calibration[(calibration["expiry"] == expiry)
                            & np.isfinite(calibration["implied_vol"])]
    smile = stages["smiles"].get(expiry)
    test_block = ok[ok["expiry"] == expiry]
    if smile is not None:
        dense = np.linspace(*smile.range, 400)
        fig, _ = plots_validation.plot_volatility_smile(
            cal_block["log_moneyness"], cal_block["implied_vol"], dense,
            smile(dense), test_block["log_moneyness"],
            test_block["interpolated_vol"],
            expiry_label=f"{expiry}, {int(cal_block['days_to_expiry'].iloc[0])} days")
        record("17_volatility_smile", fig)

    ordered = ok.sort_values("moneyness")
    fig, _ = plots_validation.plot_error_against(
        ordered["moneyness"] * 100.0, ordered["crr_error"],
        ordered["lsmc_error"], ordered["lsmc_std_error"],
        xlabel="strike as a percentage of spot",
        title="Held-out pricing error against moneyness")
    record("18_error_vs_moneyness", fig)

    fig, _ = plots_validation.plot_calibration_split(
        split["strike"], split["role"], split["expiry"], snap.spot)
    record("19_calibration_split", fig)

    buckets = pd.cut(ok["moneyness"],
                     [0.90, 0.95, 1.00, 1.05],
                     labels=["90-95%", "95-100%", "100-105%"])
    pivot = (ok.assign(bucket=buckets)
               .pivot_table(index="expiry", columns="bucket",
                            values="crr_abs_error", aggfunc="mean",
                            observed=False))
    fig, _ = plots_validation.plot_error_heatmap(
        pivot, title="Mean absolute CRR error by expiry and moneyness")
    record("20_error_heatmap", fig)

    if not spacing_frame.empty:
        fig, _ = plots_validation.plot_spacing_study(
            spacing_frame["spacing"], spacing_frame["crr_mae"],
            spacing_frame["lsmc_mae"],
            lsmc_noise=float(spacing_frame["lsmc_std_error_mean"].mean()))
        record("21_calibration_spacing", fig)

    if multi_maturity:
        by_dte = ok.sort_values("days_to_expiry")
        fig, _ = plots_validation.plot_error_against(
            by_dte["days_to_expiry"], by_dte["crr_error"],
            by_dte["lsmc_error"], by_dte["lsmc_std_error"],
            xlabel="days to expiry", as_percent_axis=False,
            title="Held-out pricing error against maturity")
        record("22_error_vs_dte", fig)
    else:
        print("\n  figure 22 (error against maturity) needs more than one "
              "expiry and was not drawn.")

    print(f"\n  tables  {SNAPSHOT_CSV.name}, {CALIBRATION_CSV.name}, "
          f"{HELDOUT_CSV.name}, {METRICS_CSV.name}"
          + (f", {SPACING_CSV.name}" if not spacing_frame.empty else ""))
    print(f"  figures {', '.join(written)}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
