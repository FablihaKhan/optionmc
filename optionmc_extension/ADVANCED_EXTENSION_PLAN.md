# Advanced Extension Plan

Four layers added on top of the completed OptionMC extension:

> **Pricing engine → validated model → real-market decision → robust risk analysis → decision dashboard**

Nothing already working is replaced. Every new capability is a new module that
calls the existing ones.

---

## 1. Frozen baseline (recorded 2026-08-19, before any new code)

| | |
|---|---|
| Branch | `feature/advanced-risk-dashboard`, cut from `project-dev` (`7e38a86`) |
| Working tree at cut | clean; nothing deleted or reset |
| Extension tests | **145 passed** |
| Base OptionMC tests | **12 passed**; `git status optionmc/` empty (package untouched) |
| Pipeline | `python main.py --skip-fetch` → **8/8 phases passed** |

### Reproducibility check

The whole pipeline was re-run and every output compared byte-for-byte against the
committed artifacts. 14 of 20 files were bit-identical. The 6 that differed did so
**only in wall-clock timing columns** (`mean_runtime_sec`, `standard_time`,
`antithetic_time`, `lsmc_runtime_sec`, `binomial_runtime_sec`).

No price, error, VaR or CVaR changed by a single bit. The seeds hold.

### SPY snapshot the whole study rests on

| Field | Value |
|---|---|
| As of | 2026-08-18 |
| Spot S0 | 768.37 |
| Strike K | 749.00 (97.5% moneyness) |
| Expiry | 2026-10-30 (73 days, T = 0.2000 yr) |
| Market put | 12.25 (source: **mid** of 12.23 / 12.27) |
| Historical sigma | 0.172149 |
| Historical mu | 0.112408 — real-world, risk only |
| Dividend yield q | 0.012085 |
| Risk-free r | 0.037874 continuous (FRED DGS3MO, 2026-08-14) |

### Headline results to preserve

| Quantity | Value |
|---|---|
| LSMC American put (historical sigma) | 13.5111 (s.e. 0.2128) |
| CRR binomial benchmark | 13.5799 — relative error 0.51% |
| Early-exercise premium | 0.1633; 22.64% of paths exercise early |
| LSMC at market-implied sigma = 0.161494 | 12.2020 vs market 12.25 (0.39%) |
| Hedge cost | $1,353.49 (1.76% of the stock position) |
| Risk reduction, % of portfolio value | 95% VaR 43.29 / 95% CVaR 48.91 / 99% VaR 51.13 / 99% CVaR 55.12 |

### Result files (20 tracked artifacts)

14 CSV tables plus 12 figures (PNG and PDF) in `results/`; `market_snapshot.json`,
`pricing_grid.json`, `portfolio_losses.npz`, `spy_history.csv`,
`spy_option_snapshot.csv`, `risk_free_dgs3mo.csv` in `data/`.

`spy_option_snapshot.csv` holds **208 strikes (470-900) for expiry 2026-10-30**
with bid/ask/volume/openInterest/IV. Addition 1 therefore needs no network.

---

## 2. The four additions

### 2.1 Protective Put Optimizer

**Question:** which real SPY put gives the best downside protection for its cost?

Candidates are the nearest listed strikes to 90.0 / 92.5 / 95.0 / 97.5 / 100.0% of
spot, duplicates removed, valid quotes required. **Ask** is the acquisition cost;
**mid** is kept separately for model-vs-market comparison.

| Reused | For |
|---|---|
| `market_data.select_put_contract` | bid/ask/mid, mid-vs-last fallback flag |
| `binomial.implied_volatility_american_put` | per-candidate sigma (CRR, never Black-Scholes) |
| `lsmc.price_american_put_lsmc` | candidate LSMC price |
| `binomial.crr_american_put` | benchmark and LSMC-vs-CRR error |
| `pricing_grid.*`, `interpolation.*` | one cached grid per strike; fast horizon repricing |
| `risk_simulation.simulate_horizon_scenarios` | **one** scenario set shared by all candidates |
| `portfolio.*`, `var_cvar.*` | Portfolio 0 / Portfolio K, VaR and CVaR in dollars and percent |

One scenario set for every candidate is deliberate: with common random numbers the
difference between two strikes is the hedge, not Monte Carlo noise.

**Genuinely new:** the balanced score and the Pareto filter.

```
protection_score = (cvar99_reduction - min) / (max - min)
cost_score       = (max_cost - cost)        / (max_cost - min_cost)
balanced         = w_protection * protection_score + w_cost * cost_score
```

Default weights 0.5 / 0.5, configurable. If every candidate scores alike the
denominator vanishes and the score is defined as 0.5 — documented, not hidden.

Four categories are reported side by side (cheapest / strongest / most efficient /
balanced). No single hedge is called "best", and a dominated hedge is never
labelled optimal.

New files: `src/hedge_optimizer.py`, `experiments/hedge_optimizer_experiment.py`,
`tests/test_hedge_optimizer.py`.

### 2.2 Real-market cross-section, out-of-sample

**Question:** does the framework generalise across strikes and expiries?

Roughly 30 / 60 / 90-day expiries, strikes spanning 90-105% moneyness, 15-30 usable
contracts. Sorted strikes are split deterministically by parity: even index
calibrates, odd index is held out.

Calibration contracts give American implied volatility from **CRR**. A PCHIP smile
is fitted over log-moneyness `log(K/S0)` from calibration points only. Held-out
strikes read their volatility off that curve and are priced with CRR and LSMC. A
held-out option's own price never enters its own prediction.

| Reused | For |
|---|---|
| `market_data.select_expiry` | called three times, one per maturity group |
| `binomial.implied_volatility_american_put` | calibration-set implied volatility |
| `binomial.crr_american_put`, `lsmc.price_american_put_lsmc` | held-out prediction |

**Two honest limits on reuse:**

* `interpolation.PricingGridInterpolator` is bound to a `PricingGrid` (a strike, a
  no-arbitrage clamp) and does **not** fit a volatility smile. The same *method* is
  reused — scipy PCHIP, already this project's chosen interpolant — through a small
  `VolatilitySmile` class. That is a new domain, not a duplicated algorithm.
* `market_data.download_option_snapshot` caches a single expiry. A sibling loader
  writes `data/spy_option_cross_section.csv`. The existing single-option snapshot is
  left exactly as it is.

Metrics (MAE, RMSE, median absolute percentage error) are reported overall and by
expiry group, with a documented minimum-price floor so a two-cent option cannot
manufacture a 400% "error".

New files: `src/cross_section_validation.py`,
`experiments/cross_section_validation_experiment.py`,
`tests/test_cross_section_validation.py`.

### 2.3 Historical bootstrap and stress testing

**Question:** does the protective-put conclusion survive when risk comes from
observed returns instead of a GBM assumption?

GBM is **not** removed. A second engine resamples 10 daily log returns with
replacement from the 1,255 cached SPY observations, sums them, and exponentiates.
No distribution is fitted, and the risk-free rate never enters.

| Reused | For |
|---|---|
| `market_data.load_spy_history` | the cached return series |
| `interpolation.PricingGridInterpolator` | the *same* grid reprices both engines |
| `interpolation.out_of_range` | counts scenarios past the grid edge |
| `portfolio.*`, `var_cvar.*`, `sanity.check_risk_measures` | unchanged |
| `binomial.crr_american_put` | deterministic put value under each shock |

**Known risk, handled up front:** bootstrap tails are fatter than GBM's, so more
scenarios may land outside the `[0.60 S0, 1.40 S0]` grid. The count is measured and
reported first; if it is non-zero the grid is widened. Nothing is silently
extrapolated.

Deterministic shocks at 0 / -5 / -10 / -20 / -30% at the 10-day horizon, priced at
the correct remaining maturity, reported both from the interpolated grid (for
consistency with the risk engine) and from CRR directly (as a check that the two
agree).

New files: `src/historical_bootstrap.py`, `src/stress_testing.py`,
`experiments/bootstrap_risk_experiment.py`, `experiments/stress_test_experiment.py`,
`tests/test_historical_bootstrap.py`, `tests/test_stress_testing.py`.

### 2.4 Streamlit decision dashboard

Seven pages: Overview, Pricing Lab, Hedge Optimizer, Market Validation, Risk and
Stress Lab, Numerical Methods, Methodology and About.

Architecture is strictly `numerical engine -> ui adapter -> streamlit`. No
mathematics is copied into a UI file; the UI imports `src/`.

Presentation Mode is the default and needs no internet: it reads the cached
`market_snapshot.json`, `pricing_grid.json`, `portfolio_losses.npz` and the CSV
tables that already exist. Live refresh happens only on an explicit click.
Expensive work runs only after a form submission, never on a slider move.

Plot colours are imported from `src/plots.py` so the dashboard and the report
figures cannot drift apart: LSMC blue, CRR orange, market aqua, reference grey.
The matplotlib figures stay; Plotly charts are additional.

`streamlit` and `plotly` are **not yet installed** and will be pinned into the
requirements files.

**One small change to existing code will be needed:** `sanity.Check` results are
currently printed but not saved, so the Numerical Methods page has no data for its
pass/fail indicators. The experiment scripts will additionally write
`results/tables/sanity_checks.csv`. This appends an output; it changes no number.

New files: `app.py`, `ui/`, page modules, `.streamlit/config.toml`,
`tests/test_app_smoke.py` (Streamlit AppTest).

---

## 3. Rules carried into every phase

1. Option pricing stays risk-neutral: drift = r - q.
2. Portfolio risk stays real-world: drift = historical mu.
3. American validation uses CRR, never Black-Scholes.
4. A held-out option's own implied volatility never predicts its own price.
5. No nested Monte Carlo; the pricing grid plus interpolation does horizon repricing.
6. No invented market data, no hard-coded conclusions.
7. Seeds stay fixed; cached snapshots are preserved.
8. Every new numerical module gets tests; every phase re-runs the full suite.
9. Modules to reuse, not rewrite: `lsmc`, `binomial`, `gbm`, `black_scholes`,
   `market_data`, `pricing_grid`, `interpolation`, `portfolio`, `risk_simulation`,
   `var_cvar`, `replication`, `plots`, `config`.

## 4. Open decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Refresh the 2026-08-18 snapshot? | **Keep it.** Every figure and table rests on it. |
| 2 | Cross-section needs one network fetch | Fetch once into a separate cache file; Additions 1 and 3 stay offline. |
| 3 | Install `streamlit` and `plotly` | Needed for Addition 4; pin the tested versions. |

## 5. Phase order

`0` baseline and branch (**done**) → `1` optimizer → `2` cross-section →
`3` bootstrap and stress → `4` UI foundation → `5A-5G` pages → `6` integration
freeze → `7` documentation.

No phase begins until the previous one passes.
