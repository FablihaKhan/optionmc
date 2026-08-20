# Five-minute classroom demonstration

The exact sequence, with the numbers to expect and the sentence to say at each
stop. Everything below runs from the cached snapshot and needs **no internet**.

## Before the room fills

```bash
cd optionmc_extension
..\.venv\Scripts\streamlit run app.py
```

It opens on **Overview** in **Cached presentation snapshot** mode. Leave it
there. Check the badge under the title reads `CACHED SNAPSHOT · 18 Aug 2026 ·
SPY · 73 DTE`.

If the projector is dim, turn on **Explain simply** in the sidebar — it adds a
plain-English reading under each set of numbers.

Cold start is about 4 seconds; after that every page is under a second. Nothing
here waits on a network.

---

## 1 · Overview — 45 seconds

**Say:** "We started from a published package that prices *European* options by
Monte Carlo. Everything on this screen is what had to be built to answer the
next question."

**Point at**, in order:

- the three cards: **Price → Protect → Decide**
- **Market snapshot**: SPY 768.37, the put we bought at strike 749, 73 days out
- **Numerical engine**: LSMC 13.5111 against the lattice's 13.5799 — 0.51 %
  apart, from two methods sharing no code
- the big number: **TAIL-RISK REDUCTION 55.12 %**

Scroll to **the argument in one row** — three panels: the unhedged loss
distribution, the put's payoff, and the protected distribution.

**Say:** "The tail on the left reaches $9,660. On the right it stops at $3,300.
The middle panel is why."

---

## 2 · Pricing Lab — 60 seconds

**Say:** "An American option can be exercised any day, so its value is not one
expectation — it is the value of an optimal stopping rule. There is no formula."

**Click** `Run pricing`. It takes about 2.5 seconds.

**Point at:**

- **Early exercise premium 0.1633**, with **22.6 % of paths exercising early** —
  the thing the base package could not price
- the verdict under the charts: *the two independent numerical methods agree to
  0.51 % of the price, inside the simulation's own two-standard-error range*

**Open the `Early exercise` tab.** The boundary rises towards the strike as
expiry approaches.

**Say:** "Below that line, exercising beats waiting. It climbs because the time
value that made waiting worthwhile is running out."

*If asked how the method works:* open **How Longstaff-Schwartz works** at the
bottom — seven steps, backwards from expiry.

---

## 3 · Hedge Optimizer — 75 seconds  ← the centrepiece

**Say:** "Knowing the price does not solve the investor's problem. Which put
should they actually buy?"

**Click** `Analyze hedges`. About 1.4 seconds: five real listed contracts, each
against 50,000 scenarios.

**Point at the frontier.** Five points, one per real SPY contract. Right is more
expensive, up is more protection.

**Then do this** — it is the moment that lands:

> Drag the **Protection weight** slider from **0.5 to 0.0**.
> The recommendation jumps to **K=692**, the cheapest.
> Drag it to **1.0**. It jumps to **K=768**, the strongest.

**Say:** "The recommendation walks along the frontier as the investor's
priorities change. There is no single best put — that is the finding, not a
dodge."

**Then point at the efficiency chart:** $2.84 → $3.00 → **$3.03** → $2.90 →
$2.57 of tail loss avoided per dollar spent.

**Say:** "Efficiency peaks in the middle at K=730. The cheapest hedge is not the
most efficient, and neither is the strongest."

---

## 4 · Market Validation — 60 seconds

**Say:** "There is a way of testing an option pricer that proves nothing. Take a
market price, solve for the volatility that reproduces it, put that volatility
back in, and report a tiny error. The model has been handed the answer."

**Point at the split chart.** Blue circle, orange diamond, blue circle, all the
way across, with blue at both ends.

**Say:** "We split the strikes. Only the blue ones were used to build the
volatility curve. The orange ones were held back and priced from their
*neighbours* — their own quotes never touched anything used to predict them."

**Point at the results:** CRR mean absolute error **$0.0188** on 11 held-out
contracts, a typical error of **0.11 %**.

**Say honestly:** "This does not show the model beats the market. The pricer
valuing a held-out contract is the same one that inverted the calibration
quotes. What it shows is that the volatility surface is smooth enough to
interpolate, and that the pipeline works on contracts it has not seen."

*If time allows:* scroll to the density study — the error is **lowest at $10
spacing**, and worse at $2. Densest is not best.

---

## 5 · Risk & Stress Lab — 60 seconds

**Say:** "We did not want the answer to depend on assuming returns are normal."

**Open the `Historical bootstrap` tab.** Scroll to the return distribution.

**Say:** "The green bars are what SPY actually did — they reach −6 % and +10 %.
The blue curve is the normal that GBM assumes; it has died out by 3.5 %. Excess
kurtosis is **+7.77**, and a normal distribution has zero."

**Open `Compare models`.**

**Say:** "The bootstrap gives a 99 % CVaR of **$6,852** against GBM's **$6,453**
— 6.2 % harsher. And the hedge cuts *more* under the bootstrap, 56.1 % against
54.3 %. Measuring risk more realistically makes the put look better, not worse."

**Open `Stress test`.**

**Say:** "No probabilities here at all. If SPY falls 30 % in ten days, the
unhedged position is down 30 %. The protected one is down **4.21 %** — the same
as at −20 %, because once the market is through the strike the loss stops
growing."

---

## 6 · Numerical Methods — 45 seconds

**Say:** "Underneath all of it is the numerical analysis."

**Point at the pipeline**, then the method cards: Monte Carlo, least-squares
regression, backward induction, binomial approximation, root finding, PCHIP
interpolation, empirical quantiles, bootstrap resampling.

**Scroll to convergence.** Fitted order **−0.613** against the theoretical
**−0.5**, over 30 independent seeds per point.

**Say:** "This matters more than any single price. An estimator converging at
the right rate is behaving; one that happens to be close at a single sample size
may be close by accident."

**Scroll to the bottom — the invariants.** **12 of 12 hold**, each naming the
file it was checked against.

**Say:** "These are re-derived every time the page loads, from the tables on
disk. A recorded verdict can outlive the numbers it was about."

---

## 7 · Closing — 15 seconds

**Say:** "In one sentence —" and read the line at the bottom of **Overview**:

> We extend European Monte Carlo option pricing to American early exercise
> (worth 0.1633 of this contract), check it against an independent lattice to
> 0.51 %, test it on 11 held-out market contracts to $0.0188, use it to hedge a
> real SPY position, cutting 99 % CVaR by 55.12 %, confirm that holds when risk
> is resampled from observed returns instead of assumed normal (6.18 % apart),
> and search 5 real listed strikes for the best protection per dollar spent.

---

## Questions you should expect

**"Why SPY and not the S&P 500 index?"**
SPY options are American-style. SPX and XSP are European, so an American pricer
would have nothing to price. It is on the About page.

**"Is the model better than the market?"**
No, and the project does not claim it. The Market Validation page says so
explicitly. What is tested is the numerics.

**"Why does the hedge lose money most of the time?"**
It does — in 63.1 % of scenarios the hedged portfolio finishes behind. That is
insurance. The point is the 1 % where it does not.

**"How do you know the LSMC is right?"**
Three ways: it reproduces the Longstaff-Schwartz paper's own eight-path worked
example exactly (0.1144), it matches eleven of the paper's Table 1 prices to
within 0.01, and it agrees with an independent lattice to 0.51 % here.

**"Is there anything wrong with it?"**
Yes, and it is documented. `src/gbm.py` expects an arithmetic drift while the
estimator supplies a mean log return, which leaves the risk simulation short by
sigma-squared-T over two. It is conservative for VaR and it was reported rather
than silently patched, because fixing it would move every published number. The
comparison page carries a drift-matched arm so the effect is measured: $41 of
the CVaR gap, against $440 from the fat tails.

**"What if the internet is down?"**
It already is, as far as this app is concerned. Presentation mode reads only
saved files, and every page is tested with the network disabled.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| A page says results are missing | `..\.venv\Scripts\python main.py --skip-fetch` |
| The dashboard will not start | Check you are in `optionmc_extension`, not the parent |
| A chart is empty | Run the phase the page names; it is written on screen |
| You accidentally switched to Live mode | Switch back to **Cached presentation snapshot**; nothing is lost |

**Do not** press *Refresh live market data* during the demonstration. It works,
and it fails gracefully, but it needs the network and it changes every number on
screen.
