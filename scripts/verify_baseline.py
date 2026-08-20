#!/usr/bin/env python
"""Verify that the untouched OptionMC baseline works in this environment.

Runs the full test suite, all four examples, and both CLI methods, with the
environment workarounds this machine needs already applied. Artifacts land in
build/verify/ so nothing is written into the package tree.

    .venv/Scripts/python scripts/verify_baseline.py           # everything
    .venv/Scripts/python scripts/verify_baseline.py --tests   # tests only

Exits 0 only if every step succeeds.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# This script re-prints output captured from child processes, and two examples
# print a literal sigma. The console codepage here is cp1252, so without this the
# runner itself would die on the very output it is reporting.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_ROOT / "optionmc"          # repo root: setup.py, tests/, examples/
EXAMPLES_DIR = PKG_DIR / "examples"
BUILD_DIR = PROJECT_ROOT / "build" / "verify"

EXAMPLES = [
    "basic_option_pricing",
    "variance_reduction",
    "parameter_sensitivity_analysis",
    "moneyness_accuracy_analysis",
]


def build_env():
    """Return an environment with this machine's required workarounds applied.

    MPLBACKEND: the system Tk install is broken (missing tk.tcl), so the default
    tkagg backend fails as soon as matplotlib builds a real figure manager.

    PYTHONUTF8/PYTHONIOENCODING: the console codepage is cp1252 and two examples
    print a literal sigma, which otherwise raises UnicodeEncodeError *after* the
    computation and files have already completed.
    """
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(label, cmd, cwd, env, results):
    """Run one step, stream a short verdict, and record the outcome."""
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True,
                          capture_output=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0 and proc.stderr:
        print("--- stderr ---", file=sys.stderr)
        print(proc.stderr.rstrip(), file=sys.stderr)
    ok = proc.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {label} (exit {proc.returncode})", flush=True)
    results.append((label, ok))
    return ok


def check_import(env):
    """Fail fast if `import optionmc` does not resolve to the installed package.

    Running from PROJECT_ROOT makes `optionmc` resolve to the repo *folder* as an
    implicit namespace package, which silently shadows the real package. Every
    step below therefore uses a cwd other than PROJECT_ROOT.
    """
    probe = (
        "import optionmc, sys;"
        "sys.exit(0) if getattr(optionmc, '__file__', None) else sys.exit(1)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=str(PKG_DIR),
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ERROR: `import optionmc` did not resolve to the installed package.\n"
              "Install it editable first:\n"
              f"    {sys.executable} -m pip install -e {PKG_DIR}", file=sys.stderr)
        return False
    return True


def report_versions(env):
    code = (
        "import sys, numpy, scipy, matplotlib, seaborn, pandas, optionmc;"
        "print(f'python      {sys.version.split()[0]}');"
        "print(f'optionmc    {optionmc.__version__}');"
        "print(f'numpy       {numpy.__version__}');"
        "print(f'scipy       {scipy.__version__}');"
        "print(f'matplotlib  {matplotlib.__version__}');"
        "print(f'seaborn     {seaborn.__version__}');"
        "print(f'pandas      {pandas.__version__}');"
        "print(f'backend     {matplotlib.get_backend()}')"
    )
    print("=" * 70)
    print(">>> environment")
    print("=" * 70)
    subprocess.run([sys.executable, "-c", code], cwd=str(PKG_DIR), env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", action="store_true", help="run only the test suite")
    parser.add_argument("--keep", action="store_true",
                        help="keep artifacts from a previous run instead of clearing them")
    parser.add_argument("--iterations", type=int, default=100000,
                        help="iteration count for the CLI checks (default: 100000)")
    args = parser.parse_args()

    env = build_env()
    if not check_import(env):
        return 1
    report_versions(env)

    if not args.keep and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Tests run from PKG_DIR so pytest.ini/conftest.py apply and the import
    # resolves correctly.
    run("pytest (12 baseline tests)", [sys.executable, "-m", "pytest"], PKG_DIR, env, results)

    if not args.tests:
        for name in EXAMPLES:
            run(f"example: {name}",
                [sys.executable, str(EXAMPLES_DIR / f"{name}.py")],
                BUILD_DIR, env, results)

        for method in ("standard", "antithetic"):
            run(f"cli: price --method {method}",
                [sys.executable, "-m", "optionmc.cli", "price",
                 "--s0", "100", "--strike", "100", "--volatility", "0.2",
                 "--time", "1.0", "--iterations", str(args.iterations),
                 "--method", method, "--seed", "42",
                 "--output-dir", f"cli_{method}"],
                BUILD_DIR, env, results)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for label, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} steps passed")
    if not args.tests:
        print(f"Artifacts: {BUILD_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
