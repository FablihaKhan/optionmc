# Extending OptionMC: American Option Pricing with Least-Squares Monte Carlo and Portfolio Risk Analysis using VaR and CVaR

An extension of the OptionMC European-option Monte Carlo package into an
American-option pricing, portfolio-risk and hedge-decision system, built on real
SPY market data and presented through an interactive dashboard.

The original OptionMC package is **used, never modified**. Every file in
`../optionmc/optionmc/` is byte-identical to the published version, and this
project calls its classes directly rather than reimplementing them.

## The three papers

| Role | Paper |
|---|---|
| Base | Herho, Kaban & Nugraha (2025), *OptionMC: A Python Package for Monte Carlo Pricing of European Options*, Int. J. Data Science 6(2), 70–84 |
| Main extension | Longstaff & Schwartz (2001), *Valuing American Options by Simulation: A Simple Least-Squares Approach*, Rev. Financial Studies 14(1), 113–147 |
| Risk reference | Rockafellar & Uryasev (2000), *Optimization of Conditional Value-at-Risk*, J. Risk 2, 21–41 |

## Project story

Each step exists because the previous one could not answer the next question.

```
Base OptionMC             European Monte Carlo: average a discounted payoff
       |                  ... but an American option can be exercised early
American LSMC             Longstaff-Schwartz: regress the continuation value
       |                  ... but how do we know the implementation is right?
CRR validation            an independent lattice, sharing no code
       |                  ... but does it work on a contract someone can buy?
Real SPY options          a listed put, its own quotes, its own implied vol
       |                  ... but what is the option actually for?
Portfolio VaR / CVaR      100 shares, 10 days, 50,000 scenarios
       |                  ... but which of the listed puts should be bought?
Protective-put optimizer  five real strikes, protection against cost
       |                  ... but does the pricing generalise beyond one quote?
Out-of-sample validation  held-out contracts priced from their neighbours
       |                  ... but does the answer survive a non-normal world?
Bootstrap and stress      resampled history, and crashes with no probability
       |                  ... but can any of this be explained in five minutes?
Decision dashboard        seven pages, offline, from cached results
```

## What is genuinely new in this extension

The base package prices European options by Monte Carlo. Everything below is
this project's own work.

- **A Longstaff-Schwartz implementation from the paper**, not a library call. It
  reproduces the paper's own eight-path worked example exactly (0.1144) and
  matches eleven of its Table 1 benchmark prices to within 0.01.
- **Three lattices** -- CRR American, European and Bermudan. The Bermudan one
  exists to separate Monte Carlo error from time-discretisation error, which are
  otherwise indistinguishable and can cancel.
- **American implied volatility** by Brent inversion of the tree, because
  inverting Black-Scholes on an American quote folds the early-exercise premium
  into sigma.
- **A pricing grid with common random numbers plus shape-preserving
  interpolation**, so 50,000 risk scenarios reprice the hedge without a nested
  Monte Carlo.
- **A protective-put optimizer** over real listed strikes, with a transparent
  scoring rule and a Pareto frontier.
- **A genuinely out-of-sample cross-section validation**, in which a held-out
  contract's own price never touches the volatility used to predict it.
- **A second risk engine** that resamples observed returns instead of assuming
  normal ones, matched to the GBM arm on mean and variance so the difference is
  attributable to the distribution's shape alone.
- **A dashboard whose presentation mode is provably offline**, tested by
  rendering every page with the network disabled.

## What it does

1. Reproduces the base OptionMC European results.
2. Downloads real SPY data and picks a listed American put (expiry 60–90 days
   out, strike at 95–100 % of spot).
3. Prices that put by **Least-Squares Monte Carlo**, implemented from the
   Longstaff–Schwartz paper — no library American pricer anywhere.
4. Validates it against a **Cox–Ross–Rubinstein binomial tree**. Black–Scholes
   is never used as the American benchmark; it has no American solution.
5. Runs three numerical experiments: convergence in paths, time discretisation,
   and regression-basis complexity.
6. Builds an American-put **pricing grid** at the risk horizon and interpolates
   it, so 50,000 risk scenarios need no nested Monte Carlo.
7. Simulates two portfolios over 10 trading days and measures 95 % / 99 % VaR
   and CVaR, and how much of each the hedge removes.
8. **Optimises the hedge** across five real listed strikes, reporting four
   objectives and a protection-cost frontier.
9. **Validates out of sample** across the option cross-section, with the
   calibration and held-out sets disjoint by construction.
10. **Re-runs the risk** under a historical bootstrap, and stress-tests the hedge
    at 0 / -5 / -10 / -20 / -30 %.
11. Presents all of it through a **seven-page Streamlit dashboard** that works
    with no internet.

## Quick start

```bash
# from the parent directory (e:\4-1\Numerical\project)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .\optionmc          # the base package
.venv\Scripts\python -m pip install -r optionmc_extension\requirements.txt

Then, **from the `optionmc_extension` directory**:

```bash
..\.venv\Scripts\python -m pytest              # 591 tests
..\.venv\Scripts\python main.py --skip-fetch   # every phase, cached data (~172 s)
..\.venv\Scripts\python main.py                # the same, refreshing market data
..\.venv\Scripts\streamlit run app.py          # the dashboard
```

On Linux or macOS, replace the backslashes with forward slashes and `Scripts`
with `bin`. Every one of these was run as written.

`main.py` runs every phase in order and stops at the first failure, because each
one depends on the last.

```bash
python main.py --list          # show the phases
python main.py --from 11       # start at the advanced extension
```

Each phase is also a standalone script:

```bash
python experiments/reproduce_optionmc.py    # phase 1
python experiments/fetch_market_data.py     # phase 5  (--refresh to re-download)
python experiments/price_spy_put.py         # phase 5
python experiments/convergence.py           # phase 6, experiment 1
python experiments/discretization_test.py   # phase 6, experiment 2
python experiments/regression_test.py       # phase 6, experiment 3
python experiments/interpolation_test.py    # phase 7
python experiments/portfolio_risk.py        # phases 8 and 9
python experiments/make_figures.py          # phase 10
python experiments/hedge_optimizer_experiment.py           # advanced 1
python experiments/cross_section_validation_experiment.py  # advanced 2
python experiments/bootstrap_risk_experiment.py            # advanced 3a
python experiments/stress_test_experiment.py               # advanced 3b
```

## The dashboard

```bash
cd optionmc_extension
..\.venv\Scripts\streamlit run app.py
```

Seven pages, all reading the results the pipeline already wrote:

| Page | The question it answers |
|---|---|
| Overview | What is this project, in thirty seconds? |
| Pricing Lab | What is an American put worth, and how was that computed? |
| Hedge Optimizer | Which real listed put should be bought? |
| Market Validation | Does the model generalise, or only reprice what it was given? |
| Risk & Stress Lab | How bad can the loss get, and does that depend on the model? |
| Numerical Methods | What methods are underneath, and what is the evidence? |
| Methodology & About | What was assumed, and how is all of it reproduced? |

**Presentation mode is the default and needs no internet.** Every number comes
from the cached snapshot and the saved tables. Live data is fetched only after
switching mode and pressing refresh, and a failed fetch leaves the cached numbers
in place rather than emptying the page. This is tested rather than asserted:
every page is rendered with `socket.connect` replaced by a function that raises.

The architecture is one-directional -- `src/` (numerical engine) to `ui/`
(adapter) to `ui_pages/` (Streamlit). Nothing under `ui/` implements mathematics.

Measured on the development machine: cold start 3.8 s, warm 0.75 s, each page
0.55-0.99 s, *Run pricing* 2.6 s at 10,000 paths across all four tabs, and
*Analyze hedges* 1.4 s for five candidates against 50,000 scenarios each.

## Numerical methods

| Method | Where it is used | Why it is needed |
|---|---|---|
| Monte Carlo simulation | every option price, every risk scenario | the only tractable route once the payoff depends on a whole path |
| Least-squares regression | the continuation value at each exercise date | it is a conditional expectation, estimated from simulated paths |
| Backward induction | the LSMC recursion and the CRR tree | an optimal stopping problem cannot be solved forwards |
| Binomial approximation | the CRR benchmark, and implied-volatility inversion | deterministic, so it carries no sampling error |
| Root finding (Brent) | implied volatility from a market price | the map from sigma to price has no closed-form inverse |
| PCHIP interpolation | the pricing grid, and the volatility smile | passes through every point without a spline's overshoot |
| Empirical quantiles | VaR, and the tail mean behind CVaR | assumes no distribution, which is the point |
| Bootstrap resampling | the historical risk engine, and the CVaR intervals | carries the data's own skew and fat tails |
| Antithetic variates | every simulation in the project | halves the paths needed for a given precision |
| Common random numbers | the pricing grid, and the hedge comparison | makes two candidates differ by the hedge, not by the draw |

## Layout

```
optionmc_extension/
├── main.py                  runs every phase in order
├── config.py                every parameter, in one place
├── requirements.txt
├── data/                    cached market data (never re-downloaded silently)
│   ├── spy_history.csv          SPY daily prices
│   ├── spy_option_snapshot.csv  the put chain slice
│   ├── risk_free_dgs3mo.csv     FRED 3-month Treasury
│   ├── market_snapshot.json     the frozen inputs every phase reads
│   ├── pricing_grid.json        the American-put grid at the horizon
│   └── portfolio_losses.npz     the two loss distributions
├── src/
│   ├── black_scholes.py     European benchmark, with dividend yield q
│   ├── gbm.py               full-path GBM; the two measures kept separate
│   ├── binomial.py          CRR American / European / Bermudan trees
│   ├── lsmc.py              Longstaff–Schwartz — the core extension
│   ├── european_mc.py       calls the ORIGINAL OptionMC for phase 1
│   ├── market_data.py       yfinance + FRED, with caching and selection rules
│   ├── replication.py       repeated runs and convergence-order fitting
│   ├── pricing_grid.py      the grid, with common random numbers
│   ├── interpolation.py     linear / cubic / pchip, with clamping
│   ├── risk_simulation.py   real-world horizon scenarios
│   ├── portfolio.py         the two portfolios and their losses
│   ├── var_cvar.py          VaR, CVaR and the Rockafellar–Uryasev form
│   ├── sanity.py            the checks the scope requires
│   └── plots.py             the figures
│   ├── hedge_optimizer.py   candidates, scoring and the Pareto frontier
│   ├── cross_section_validation.py  the out-of-sample split and the smile
│   ├── historical_bootstrap.py      the second risk engine
│   ├── stress_testing.py    deterministic crash scenarios
│   └── plots*.py            the report figures
├── ui/                      the dashboard adapter -- no mathematics here
│   ├── compute.py           cached calls into src/
│   ├── data_loader.py       reads saved results; the only network path
│   ├── components.py        hero, cards, callouts, check lists
│   ├── interactive_charts.py  Plotly, sharing the report's palette
│   ├── explanations.py      plain-English readings, built from values
│   ├── formatters.py        one place where a dollar is written
│   └── state.py             session state, shared across pages
├── ui_pages/                one module per dashboard page
├── app.py                   the dashboard entry point
├── .streamlit/config.toml   the dashboard theme
├── experiments/             one script per phase, 13 in total
├── tests/                   591 tests across 18 files
└── results/
    ├── tables/              28 tables, every number as CSV
    └── figures/             26 figures, PNG + PDF at 300 dpi
```

## Results

Market snapshot, 18 August 2026 (cached in `data/`):

| | |
|---|---|
| SPY spot S₀ | 768.37 |
| Strike K | 749.00 (moneyness 0.9748) |
| Expiry | 2026-10-30, 73 days, T = 0.2000 yr |
| Market put price | 12.25 (bid 12.23 / ask 12.27, mid) |
| Historical σ | 0.1721 |
| Historical μ | 0.1124 — **real world, risk simulation only** |
| Dividend yield q | 0.0121 |
| Risk-free r | 0.0379 (FRED DGS3MO 3.86 %) |

### American put price

| Method | Price |
|---|---|
| LSMC (10,000 paths × 50 steps, degree 2) | 13.5111 (s.e. 0.2128) |
| CRR binomial, 2000 steps | 13.5799 |
| Black–Scholes European | 13.4141 |
| Market | 12.2500 |

LSMC agrees with the binomial to 0.51 %. The early-exercise premium is 0.1633
and 22.6 % of paths exercise before expiry — the feature this extension exists
to price. The model sits above the market because the historical volatility
(17.21 %) exceeds the market's implied volatility (16.15 %, inverted from the
quote against the American tree): a volatility risk premium, not a pricing
error.

### Tail risk over 10 trading days, 50,000 scenarios

As a percentage of each portfolio's own initial value:

| Portfolio | 95 % VaR | 95 % CVaR | 99 % VaR | 99 % CVaR |
|---|---|---|---|---|
| 100 SPY | 5.1435 % | 6.4764 % | 7.2861 % | 8.3986 % |
| 100 SPY + 1 put | 2.9168 % | 3.3088 % | 3.5611 % | 3.7694 % |
| **reduction** | **43.29 %** | **48.91 %** | **51.13 %** | **55.12 %** |

The hedge costs 1,353.49 up front (1.76 % of the share position) and removes
3,505.96 of 99 % CVaR — 2.59 dollars of extreme tail loss avoided per dollar
spent. It is not free: in 63.1 % of scenarios the hedged portfolio finishes
behind the unhedged one, which is the premium being paid.

CVaR improves more than VaR at both levels: the put bites hardest in the far
tail, which is what a protective put is bought to do.

### Which put to buy (five real listed strikes)

Cost is the **ask**, which is what a buyer pays; each contract carries the
volatility implied by its own quote.

| Strike | Cost % | 99 % VaR red. | 99 % CVaR red. | CVaR saved / $ |
|---|---|---|---|---|
| 692 (90.1 %) | 0.61 | 17.73 | 20.49 | 2.84 |
| 711 (92.5 %) | 0.81 | 25.56 | 28.97 | 3.00 |
| 730 (95.0 %) | 1.12 | 36.45 | 40.44 | **3.03** |
| 749 (97.5 %) | 1.60 | 50.93 | 55.13 | 2.90 |
| 768 (100 %) | 2.35 | 68.14 | **71.81** | 2.57 |

Cheapest 692, strongest 768, most efficient 730, balanced 749. Efficiency turns
over inside the range: the cheapest hedge is not the most efficient one, and
neither is the strongest. All five are Pareto efficient, so none is dominated
and none is called "best".

The calibrated volatilities run 0.2183 at the 90 % strike to 0.1453 at the
money -- the put skew, priced from real quotes rather than assumed.

### Out-of-sample validation (11 held-out contracts)

| Model | MAE | RMSE | Median error |
|---|---|---|---|
| CRR binomial | 0.0188 | 0.0264 | 0.11 % |
| LSMC | 0.0724 | 0.0985 | 0.41 % |

All 11 LSMC errors fall inside that contract's own two standard errors, so the
simulation's disagreement with the market is its own sampling noise rather than
a modelling failure.

What this shows: the American implied-volatility surface is smooth enough that a
strike left out of the fit can be repriced from its neighbours to within a few
cents. What it does **not** show: that the model beats the market. The pricer
valuing a held-out contract is the same one that inverted the calibration
quotes, so what is tested is the interpolation and the pipeline, not a view.

### Risk model comparison

| Engine | 99 % CVaR, unhedged | Hedge reduction |
|---|---|---|
| GBM Monte Carlo | 6,453.24 | 54.33 % |
| Historical bootstrap | **6,852.33** | 56.10 % |

The bootstrap is 6.18 % harsher. Observed daily returns have excess kurtosis
+7.77; a ten-day fall worse than -10 % is 1.87x as likely under it, and a fall
worse than -15 % never occurred in 50,000 GBM draws but did under the bootstrap.
The hedge reduces tail loss under both engines -- and by *more* under the
bootstrap, so measuring risk more realistically makes the put look better rather
than worse.

### Stress tests

| Shock | SPY | Stock loss | Protected loss | Hedge benefit |
|---|---|---|---|---|
| 0 % | 768.37 | 0.00 % | 0.26 % | -200.98 |
| -5 % | 729.95 | 5.00 % | 2.86 % | +1,602.57 |
| -10 % | 691.53 | 10.00 % | 4.03 % | +4,536.38 |
| -20 % | 614.70 | 20.00 % | 4.21 % | +12,076.91 |
| -30 % | 537.86 | 30.00 % | **4.21 %** | +19,760.61 |

The protected loss flattens once the market is through the strike. The put does
not remove the loss; it caps how fast it grows. In a flat market it costs 200.98
of time value -- what protection costs when it turns out not to be needed.

## How the results are verified

Validation is anchored to **published numbers**, not to self-consistency.

**The Longstaff–Schwartz worked example.** Section 1 of the paper prices an
American put on eight hand-written paths. Our implementation reproduces its
value of **0.1144** (and the European 0.0564), its stopping rule, and its
regression coefficients −1.070 + 2.983X − 1.813X² exactly. That fixture is in
`tests/test_lsmc.py`.

**Table 1 of the same paper.** Our CRR tree matches the paper's finite-difference
American values to ≤ 0.01 across all 11 parameter sets, and our LSMC matches to
about one standard error:

| S | σ | T | paper FD | our CRR | paper LSM | our LSMC |
|---|---|---|---|---|---|---|
| 36 | 0.20 | 1 | 4.478 | 4.487 | 4.472 | 4.482 (0.009) |
| 40 | 0.20 | 1 | 2.314 | 2.320 | 2.313 | 2.308 (0.009) |
| 44 | 0.40 | 2 | 5.647 | 5.647 | 5.622 | 5.612 (0.021) |

**Rockafellar–Uryasev Theorem 1**, used as a live check rather than a citation.
CVaR computed by minimising F<sub>β</sub>(α) = α + (1−β)⁻¹E[(L−α)⁺] matches the
empirical tail mean to the cent, and the minimiser comes back as the VaR:

```
SPY only  99%:  min F = 6,453.24   empirical CVaR = 6,453.24   alpha* = 5,598.81   VaR = 5,598.41
```

**591 tests**, including the scope's sanity checks: American ≥ European,
American ≥ intrinsic, American ≤ K, price settles as paths increase, benchmark
settles as tree steps increase, CVaR ≥ VaR.

**71 whole-project integration checks** for the things that fall between files
and that nothing raises on: a nan reaching a metric card or a chart axis, a
chart with traces but no data, a chart disagreeing with the table it was drawn
from, money written without a thousands separator, a page that does not say
which snapshot it is using. The frontier, the smile, the stress curve and the
convergence series are each compared value by value against their saved CSV.

**12 invariants re-derived on every dashboard load** rather than read from a
stored verdict, each naming the file it was checked against: American ≥
European, American ≥ intrinsic, the put falling as the underlying rises,
CVaR ≥ VaR, the measures separated, no calibration leakage, no extrapolated
volatility, the convergence rate, the interpolation error, no dominated point on
the frontier, and a signature check that the bootstrap engine cannot be handed a
risk-free rate. A test breaks one of the saved tables and requires the
corresponding invariant to turn red, which is what proves they are computed and
not recalled.

**The base package is unmodified**: `git status optionmc/` is empty, its 12
tests pass, and its verification script runs 7/7 including four examples and
both CLI methods.

## Findings from the numerical experiments

**Convergence.** The run-to-run spread of the LSMC price follows the theoretical
rate almost exactly — fitted N<sup>−0.520</sup> against a theory of N<sup>−0.5</sup>
(R² = 0.98). RMSE falls faster, N<sup>−0.613</sup>, because the small-sample bias
is decaying at the same time.

**The LSMC bias changes sign.** At 1,000 paths the price is biased *upward*
(+0.51): a regression fitted on few in-the-money paths partly fits noise, and
the exercise rule then acts on information it does not really have. At 50,000
paths the bias is *downward* (−0.012), the residue of an imperfect exercise
rule. So |bias| is not monotone in N, and a sanity check demanding that it were
would fail for an entirely legitimate reason.

**Time discretisation.** Going from 10 to 100 exercise dates shrinks the
discretisation error nine-fold (0.0336 → 0.0037) and costs 9.6× the runtime,
while the Monte Carlo error stays near 0.05 throughout. More steps do not fix
Monte Carlo error; only more paths do. A caution for readers of that table: the
*total* error looks smallest at 10 steps, but that is two opposite errors
cancelling by coincidence, not accuracy — judge the two columns separately.

**A higher regression degree is not automatically better.** At 50,000 paths the
biases are −0.064 (degree 1), −0.005 (degree 2), +0.031 (degree 3): the linear
basis underfits the continuation value and exercises suboptimally, the cubic
fits sampling noise. Degree 2 is the turning point. The comparison is made over
30 independent seeds per cell with a two-sample z statistic, because at 10,000
paths degrees 1 and 2 are *not* statistically distinguishable (z = 0.89) and
claiming a winner there would be reading noise.

**Calibration density is not monotone.** Held-out error is lowest at $10
strike spacing (0.0188), worse at $40 (0.0561) where the curve has too far to
reach, and *also* worse at $2 (0.0218) where PCHIP threads the penny-wide
quantisation of the bid-ask quotes and carries that wobble into the held-out
strikes. The held-out set is held fixed across the sweep so only the calibration
grid changes. Densest is not best.

**Fat tails, isolated.** The bootstrap and a drift-matched GBM share a mean and
a standard deviation by construction, so the entire 99 % CVaR gap is the shape
of the distribution. Decomposed: $41 of the difference is a drift convention
(below), $440 is the tails.

**The pricing grid's error budget**, decomposed rather than lumped together:
interpolation 0.003 (pchip), Monte Carlo bias at the nodes 0.087, time
discretisation 0.016. The interpolation is not the bottleneck, so refining the
grid would not help — a better regression basis does, which is why the grid uses
degree 3 while the headline pricing uses the scope's degree 2.

## Design decisions worth knowing

**The two measures are kept apart by construction.** Option pricing uses the
risk-neutral drift r − q; portfolio VaR uses the historical μ. The scope names
this as the most likely and most damaging mistake, and it is dangerous precisely
because the code would run perfectly either way. So the drift is always an
explicit argument, the risk simulation's parameter is named `real_world_drift`,
and `sanity.check_measure_separation` refuses to run the risk phase if μ ever
equals r or r − q.

**No nested Monte Carlo.** Valuing the hedge in all 50,000 scenarios by direct
LSMC would need 10 billion paths. The grid takes 33 seconds. GBM paths scale
exactly with the starting price, so one set of unit paths is simulated once and
rescaled for every grid node — which is both fast and, because every node then
sees identical randomness, free of the node-to-node jitter that an interpolant
would otherwise mistake for signal.

**pchip for the interpolation.** A natural cubic spline scores slightly better
on error (0.0011 vs 0.0033) but can overshoot; pchip is shape-preserving, so it
cannot produce a put that grows more valuable as the underlying rises. Values
are clamped into max(K − S, 0) ≤ P ≤ K, which also handles anything landing
outside the grid.

**Implied volatility is computed here, not taken from the feed.** The chain's
`impliedVolatility` field reported 0.1506, which reprices the put at 10.77
against a 12.25 market — stale and inconsistent with its own quote. We invert
the observed price ourselves, against the **American** tree, because inverting
the European formula on an American quote biases the answer by the
early-exercise premium.

**The ask, not the mid, is the cost of a hedge.** An investor buying
protection pays the ask. The mid is kept alongside for model-versus-market
comparison, but every cost figure in the optimizer is what the hedge would
actually cost to put on.

**Each candidate carries its own volatility**, inverted from its own market
quote against the CRR tree. Pricing every strike off a single sigma would
quietly misprice the cheap wings, because the skew is real.

**A known inconsistency, reported rather than silently fixed.** `src/gbm.py`
simulates `S_T = S0 exp((drift - sigma^2/2) T + sigma sqrt(T) Z)`, so the drift
it expects is the arithmetic one, while `estimate_gbm_parameters` returns the
mean *log* return. Passing the second into the first leaves the simulated mean
log return short by sigma^2 T / 2 = 0.000588 over this horizon. It is small and
conservative for VaR, so nothing published was changed -- changing it would move
every number in the project. A third drift-matched arm is reported instead,
which decomposes the CVaR gap and lets the shape claim stand on its own.

**Relative error is not reported where the price is near zero.** Deep out of the
money the true put price is a fraction of a cent, so a harmless 0.0008 absolute
error reads as 4.8 %. Accuracy checks are stated in dollars and in impact on the
100-share position; relative error is shown as a profile by price floor.

## Reproducibility

Everything is seeded from `config.SEED` and every downloaded input is cached, so
a rerun reproduces the same numbers rather than picking up a moved market. This
was verified rather than assumed: re-running the whole pipeline leaves 14 of 20
data artifacts byte-identical, and the six that differ do so only in wall-clock
timing columns. Eleven of the twelve original figures are pixel-identical; the
exception is the one that plots runtime.

Rerun against fresh data with:

```bash
python experiments/fetch_market_data.py --refresh
python main.py --skip-fetch
```

## Environment notes

Two machine-specific workarounds, applied automatically by `main.py`:

- `MPLBACKEND=Agg` — the system Tk install is incomplete, so matplotlib must
  stay headless.
- `PYTHONUTF8=1` — the console codepage is cp1252 and raises on the Greek
  letters in the output.

Do not run Python with the parent directory as the working directory: the
`optionmc` folder there shadows the installed package as an implicit namespace
package.

## Scope and limitations

Deliberately fixed, and not expanded: SPY only (its ETF options are
American-style; SPX and XSP are European and would defeat the purpose), one
American put, 100 shares against one contract, a 10-trading-day horizon,
95 % and 99 % confidence, geometric Brownian motion.

What this project does not claim:

- **GBM is a simplified market model** -- constant volatility, no jumps, no
  stochastic volatility, a single risk factor.
- **The bootstrap can only resample the history it was given**, and drawing days
  independently discards volatility clustering, so a multi-day crash is less
  likely under it than in the market.
- **Market quotes can be stale or illiquid.** The snapshot is one moment, and
  options are not equally traded across strikes.
- **One expiry has live quotes in the cached chain**, so the cross-section
  validates across strikes rather than across maturities. The three-maturity
  path is implemented and needs one fetch during trading hours; outside them the
  feed returns a zero bid and ask on every contract, and the downloader refuses
  to cache that rather than validating against prices that do not exist.
- **Transaction costs beyond the bid-ask spread are not modelled** -- no
  commissions, financing or slippage.
- **Model parameters drift through time.** A volatility estimated over five
  years is not a forecast of the next ten days.
- **The hedge is static**, bought once and held to the horizon, with a loss
  horizon shorter than the option's life.

This is an educational numerical-analysis project. It is not financial advice,
not a recommendation to trade, and not a production risk system. Every number is
conditional on a single frozen market snapshot and on the assumptions above.
