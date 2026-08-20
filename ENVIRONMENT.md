# Development Environment

Setup for the OptionMC extension work. The upstream package in `optionmc/` is
treated as read-only: everything here is additive.

## Layout

```
project/
├── .venv/                    # virtual environment (CPython 3.12.10)
├── requirements.txt          # runtime deps, floors mirror optionmc/setup.py
├── requirements-dev.txt      # runtime + pytest/pytest-cov
├── requirements-lock.txt     # exact verified versions
├── ENVIRONMENT.md            # this file
├── scripts/
│   └── verify_baseline.py    # runs tests + examples + CLI with the right env
├── build/verify/             # generated artifacts (git-ignored)
└── optionmc/                 # the package repo -- upstream code untouched
    ├── conftest.py           # ADDED: forces Agg backend, closes figures
    ├── pytest.ini            # ADDED: testpaths, markers
    ├── optionmc/             # models.py, visualization.py, utils.py, cli.py
    ├── tests/                # 12 baseline tests
    └── examples/             # 4 baseline examples
```

## Setup from scratch

```powershell
cd e:\4-1\Numerical\project
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .\optionmc
```

## Verify

```powershell
.venv\Scripts\python scripts\verify_baseline.py            # tests + examples + CLI
.venv\Scripts\python scripts\verify_baseline.py --tests    # tests only
```

Baseline status as of 2026-08-19: **7/7 steps pass** (12/12 tests, 4/4 examples,
both CLI methods).

## Running things by hand

```powershell
cd e:\4-1\Numerical\project\optionmc     # note the cwd -- see gotcha 1
..\.venv\Scripts\python -m pytest
..\.venv\Scripts\python -m pytest tests/test_models.py -v
```

## Environment gotchas

Three machine-specific traps, all handled by the scaffolding above. None is a
defect in the OptionMC source.

### 1. Never run Python from the project root

`e:\4-1\Numerical\project\optionmc\` has no `__init__.py`, so with the project
root as the working directory `import optionmc` resolves to that **folder** as an
implicit namespace package and silently shadows the installed package:

```
>>> import optionmc; optionmc.__file__
None                       # namespace package -- no models, no version
```

Work from `optionmc/` (or any directory that is not the project root).
`verify_baseline.py` sets an appropriate cwd for every step and fails fast with a
clear message if the import is shadowed.

### 2. Broken Tk -> use the Agg backend

The system Python has an incomplete Tk install:

```
_tkinter.TclError: Can't find a usable tk.tcl in the following directories: ...
```

matplotlib's default backend here is `tkagg`, which tries to create a real window
once a figure manager is built. The baseline tests never close their figures, so
during a full-suite run figures accumulate and `test_plot_price_distribution`
fails intermittently -- it passes in isolation, which makes this look flaky.

Handled by `optionmc/conftest.py`: `matplotlib.use("Agg", force=True)` plus an
autouse fixture calling `plt.close("all")` after each test. Set `MPLBACKEND=Agg`
for anything that runs outside pytest.

### 3. cp1252 console -> force UTF-8

`examples/variance_reduction.py` and `examples/parameter_sensitivity_analysis.py`
print a literal `σ`. On this cp1252 console that raises `UnicodeEncodeError`
*after* the computation and file writes have already finished, so the exit status
lies about what actually happened. Set `PYTHONUTF8=1` (`verify_baseline.py` does
this, and also reconfigures its own stdout, since it re-prints child output).

## Known pre-existing bug (not ours)

`optionmc/cli.py` `price_options()` reuses `call_price`, `put_price`, `bs_call`
and `bs_put` as loop temporaries inside the convergence and volatility-sensitivity
loops. The values it finally prints and writes to `results.json` therefore come
from the last sensitivity point (`sigma = min(0.8, 2*sigma)`), not from the
parameters the user asked for:

```
$ optionmc price --s0 100 --strike 100 --volatility 0.2 --time 1.0
Call Option Price: $17.8172 (Analytical: $18.0230)     <- these are the sigma=0.4 values
                                                        true sigma=0.2: $10.4506 / $5.5735
```

The `*_error` fields in `results.json` are still the correct-parameter metrics, so
the file contradicts itself. The library in `models.py` is unaffected -- only the
CLI's reporting. Left as-is pending a decision on whether to fix it.
