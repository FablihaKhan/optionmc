#!/usr/bin/env python
"""Run the whole study, phase by phase, in the order the scope lays out.

    python main.py                 # everything
    python main.py --skip-fetch    # reuse the cached market data
    python main.py --from 6        # start at phase 6
    python main.py --list          # show the phases and stop

Each phase is a standalone script under experiments/ and can be run on its own;
this just runs them in order and reports which ones passed. The run stops at
the first failure, because every later phase depends on the earlier ones.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

PHASES = [
    (1, "Reproduce base OptionMC", "reproduce_optionmc.py", []),
    (5, "Fetch SPY market data", "fetch_market_data.py", []),
    (5, "Price the SPY American put", "price_spy_put.py", []),
    (6, "Experiment 1: convergence", "convergence.py", []),
    (6, "Experiment 2: time discretisation", "discretization_test.py", []),
    (6, "Experiment 3: regression degree", "regression_test.py", []),
    (7, "Pricing grid and interpolation", "interpolation_test.py", []),
    (8, "Portfolio simulation, VaR and CVaR", "portfolio_risk.py", []),
    (10, "Figures", "make_figures.py", []),
    # --- advanced extension -------------------------------------------------
    (11, "Advanced 1: protective put optimizer", "hedge_optimizer_experiment.py", []),
    (12, "Advanced 2: out-of-sample cross-section", "cross_section_validation_experiment.py", []),
    (13, "Advanced 3a: GBM against historical bootstrap", "bootstrap_risk_experiment.py", []),
    (14, "Advanced 3b: deterministic stress tests", "stress_test_experiment.py", []),
]


def build_environment():
    """Environment the phases need on this machine.

    MPLBACKEND: the system Tk install is broken, so matplotlib must stay
    headless. PYTHONUTF8: the console codepage is cp1252 and would raise on
    the Greek letters the output uses.
    """
    import os

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_phase(number, title, script, extra, env, quiet):
    command = [sys.executable, str(ROOT / "experiments" / script), *extra]
    banner = f"PHASE {number}  {title}"
    print(f"\n{'#' * 78}\n# {banner}\n#   experiments/{script}\n{'#' * 78}",
          flush=True)

    start = time.perf_counter()
    if quiet:
        proc = subprocess.run(command, env=env, text=True, capture_output=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
    else:
        proc = subprocess.run(command, env=env)
    elapsed = time.perf_counter() - start

    ok = proc.returncode == 0
    print(f"\n[{'PASS' if ok else 'FAIL'}] {banner}  ({elapsed:.1f}s)",
          flush=True)
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="reuse cached market data instead of downloading")
    parser.add_argument("--from", dest="start_phase", type=int, default=0,
                        help="start from this phase number")
    parser.add_argument("--quiet", action="store_true",
                        help="only show each phase's verdict, not its output")
    parser.add_argument("--list", action="store_true",
                        help="list the phases and exit")
    args = parser.parse_args()

    if args.list:
        print("Phases:")
        for number, title, script, _ in PHASES:
            print(f"  {number:>2}  {title:<38} experiments/{script}")
        return 0

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    env = build_environment()
    selected = [p for p in PHASES if p[0] >= args.start_phase]
    if args.skip_fetch:
        selected = [p for p in selected if p[2] != "fetch_market_data.py"]

    print("=" * 78)
    print("Extending OptionMC: American option pricing with Least-Squares")
    print("Monte Carlo and portfolio risk analysis using VaR and CVaR")
    print("=" * 78)
    print(f"  seed {config.SEED}, {config.N_RISK_SCENARIOS:,} risk scenarios, "
          f"{config.RISK_HORIZON_DAYS}-day horizon")
    print(f"  running {len(selected)} phases")

    results = []
    total = 0.0
    for number, title, script, extra in selected:
        ok, elapsed = run_phase(number, title, script, extra, env, args.quiet)
        results.append((number, title, ok, elapsed))
        total += elapsed
        if not ok:
            print(f"\nStopping: phase {number} failed, and the phases after it "
                  "depend on its output.")
            break

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for number, title, ok, elapsed in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] phase {number:>2}  {title:<40} "
              f"{elapsed:>7.1f}s")
    passed = sum(1 for *_, ok, _ in results if ok)
    print(f"\n  {passed}/{len(results)} phases passed in {total:.1f}s")

    if passed == len(selected):
        print(f"\n  tables  {config.TABLES_DIR}")
        print(f"  figures {config.FIGURES_DIR}")
        print(f"  data    {config.DATA_DIR}")
    return 0 if passed == len(results) == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
