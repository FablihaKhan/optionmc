"""Plain-language readings of whatever the numbers happen to be.

Every sentence here is assembled from calculated values by ordinary Python.
Nothing calls a language model at runtime: the same inputs always produce the
same words, which is what makes the explanations reproducible and safe to show
a class.

They also have to be able to say bad news. A sentence that always reads "the
two methods agree closely" is not an explanation, it is decoration -- so each
function branches on the number it was given and says something different when
the number is poor.
"""
from .formatters import (fraction_as_percent, money, percent, price,
                         signed_money, volatility)


def _drop_empty(lines):
    return [line for line in lines if line]


def pricing(lsmc=None, binomial=None, market=None, std_error=None,
            early_premium=None, exercise_fraction=None):
    """How to read an LSMC price against its benchmark and the market."""
    lines = []

    if lsmc is not None and binomial is not None and binomial:
        gap = abs(lsmc - binomial)
        relative = gap / binomial * 100
        within = (std_error is not None and std_error > 0
                  and gap <= 2 * std_error)
        if relative < 1.0:
            verdict = (f"agree to {percent(relative)} of the price"
                       + (", inside the simulation's own two-standard-error "
                          "range" if within else ""))
        elif relative < 3.0:
            verdict = (f"differ by {percent(relative)}, which is larger than "
                       "the usual sampling noise and worth a second look")
        else:
            verdict = (f"differ by {percent(relative)} — too far apart to "
                       "treat either as settled")
        lines.append(
            f"Two independent methods price this put: a simulation (LSMC, "
            f"{price(lsmc)}) and a lattice (CRR binomial, {price(binomial)}). "
            f"They {verdict}. Agreement between methods that share no code is "
            f"the main evidence the implementation is right.")

    if lsmc is not None and market is not None and market:
        gap = lsmc - market
        direction = "above" if gap > 0 else "below"
        lines.append(
            f"The model sits {price(abs(gap))} {direction} the quoted market "
            f"price of {price(market)} ({percent(abs(gap) / market * 100)}). "
            f"A gap here is not automatically an error: the model is run at "
            f"the volatility SPY actually realised, while the market prices in "
            f"the volatility it expects.")

    if early_premium is not None and exercise_fraction is not None:
        lines.append(
            f"The right to exercise early is worth {price(early_premium)} of "
            f"this price, and {fraction_as_percent(exercise_fraction)} of the "
            f"simulated paths take it. That premium is exactly what the "
            f"European Monte Carlo of the base project could not capture.")

    return _drop_empty(lines)


def hedging(cvar_reduction=None, cost_percent=None, saved_per_dollar=None,
            worse_fraction=None):
    """What a protective put did to the tail, and what it cost."""
    lines = []

    if cvar_reduction is not None:
        if cvar_reduction > 0:
            lines.append(
                f"In the worst 1% of ten-day outcomes the hedge removes "
                f"{percent(cvar_reduction)} of the average loss. That is the "
                f"number the put is bought for; it is calculated from the "
                f"simulated distribution, not assumed.")
        else:
            lines.append(
                f"Over these scenarios the hedge did not reduce tail loss "
                f"({percent(cvar_reduction)}). Protection that costs more than "
                f"it saves is a real outcome and is reported as one.")

    if cost_percent is not None:
        lines.append(
            f"It costs {percent(cost_percent)} of the share position to put "
            f"on. That premium is paid whether or not the market falls, and in "
            f"a flat market it is a pure loss.")

    if saved_per_dollar is not None:
        comparison = ("more than" if saved_per_dollar > 1 else "less than")
        lines.append(
            f"Each dollar of premium buys {money(saved_per_dollar)} of avoided "
            f"tail loss — {comparison} a dollar back. This is not a forecast "
            f"of profit: it is a ratio between a certain cost and an average "
            f"over the worst outcomes only.")

    if worse_fraction is not None:
        lines.append(
            f"In {fraction_as_percent(worse_fraction)} of scenarios the hedged "
            f"portfolio finishes behind the unhedged one. Insurance usually "
            f"does; the point is what happens in the minority where it does not.")

    return _drop_empty(lines)


def validation(mae=None, rmse=None, n_calibration=None, n_heldout=None,
               median_pct=None, mean_quote=None):
    """Why a held-out test says more than repricing an option with its own IV."""
    lines = []

    if n_calibration is not None and n_heldout is not None:
        lines.append(
            f"{n_calibration} contracts were used to build the volatility "
            f"curve and {n_heldout} were held back. Each held-back contract "
            f"was priced from its neighbours' volatility — its own market "
            f"price never touched anything used to predict it.")

    if mae is not None:
        scale = (f", against an average quote of {money(mean_quote)}"
                 if mean_quote else "")
        quality = ("close" if (mean_quote and mae < 0.01 * mean_quote)
                   else "reasonable" if (mean_quote and mae < 0.05 * mean_quote)
                   else "loose")
        lines.append(
            f"Average error on the held-out contracts is {money(mae, 4)}"
            f"{scale} — a {quality} fit. "
            + (f"Root-mean-square error {money(rmse, 4)} is larger than the "
               f"average, so a few contracts carry most of the miss."
               if rmse and rmse > 1.4 * mae else
               "The two error measures are close, so no single contract "
               "dominates the result."))

    if median_pct is not None:
        lines.append(
            f"The typical held-out contract is priced within "
            f"{percent(median_pct)} of its quote. The median is used rather "
            f"than the mean because one cheap option can distort a percentage "
            f"average beyond recognition.")

    return _drop_empty(lines)


def risk_measures(var=None, cvar=None, level=0.99, portfolio_value=None):
    """The difference between VaR and CVaR, said once, with this page's numbers."""
    lines = []
    label = f"{level:.0%}"

    if var is not None:
        share = (f" ({percent(var / portfolio_value * 100)} of the position)"
                 if portfolio_value else "")
        lines.append(
            f"{label} VaR of {money(var)}{share} answers *where the bad tail "
            f"begins*: on {fraction_as_percent(1 - level, 0)} of ten-day "
            f"outcomes the loss is at least this large.")

    if cvar is not None:
        lines.append(
            f"{label} CVaR of {money(cvar)} answers a harder question — "
            f"*once we are already in that worst {fraction_as_percent(1 - level, 0)}, "
            f"how bad is it on average*. It is always the larger of the two, "
            f"and it is the one that notices how far the tail actually reaches.")

    if var is not None and cvar is not None and var:
        lines.append(
            f"The gap between them, {money(cvar - var)}, is how much worse "
            f"things get beyond the threshold. A thin-tailed model keeps that "
            f"gap small; a fat-tailed one does not.")

    return _drop_empty(lines)


def risk_models(gbm_cvar=None, bootstrap_cvar=None, excess_kurtosis=None,
                tail_ratio=None):
    """Which risk engine is harsher here, and why."""
    lines = []

    if gbm_cvar is not None and bootstrap_cvar is not None:
        harsher, gentler = (("historical bootstrap", "GBM")
                            if bootstrap_cvar > gbm_cvar
                            else ("GBM", "historical bootstrap"))
        gap = abs(bootstrap_cvar - gbm_cvar) / min(bootstrap_cvar, gbm_cvar) * 100
        lines.append(
            f"The more conservative 99% CVaR in this snapshot comes from the "
            f"{harsher} ({money(max(gbm_cvar, bootstrap_cvar))} against "
            f"{money(min(gbm_cvar, bootstrap_cvar))} from the {gentler}, "
            f"{percent(gap)} apart). Which model wins is read off the "
            f"calculation, not decided in advance.")

    if excess_kurtosis is not None:
        lines.append(
            f"Observed SPY days have excess kurtosis of {excess_kurtosis:+.1f}. "
            f"A normal distribution has zero, so extreme days happen far more "
            f"often than GBM allows — and a protective put pays off precisely "
            f"on those days.")

    if tail_ratio is not None and tail_ratio > 0:
        lines.append(
            f"A ten-day fall worse than 10% is {tail_ratio:.1f} times as likely "
            f"under the bootstrap as under GBM. The two engines are matched on "
            f"average and on spread; they disagree only about the tail.")

    return _drop_empty(lines)


def stress(worst_unhedged=None, worst_protected=None, flat_cost=None,
           largest_benefit=None, largest_shock=None):
    """What the crash table shows, in one reading."""
    lines = []

    if worst_unhedged is not None and worst_protected is not None:
        lines.append(
            f"At the deepest shock the unprotected position loses "
            f"{percent(worst_unhedged)} while the protected one loses "
            f"{percent(worst_protected)}. The put does not remove the loss — "
            f"it caps how fast it grows once the market is through the strike.")

    if flat_cost is not None:
        # `flat_cost` arrives as a hedge benefit, so a cost is negative.
        # "costs -$200.98" is a double negative; the verb carries the sign.
        if flat_cost < 0:
            lines.append(
                f"If the market does not move at all, the hedge still costs "
                f"{money(abs(flat_cost))} of portfolio value over the ten "
                f"days. That is the option's time value decaying, and it is "
                f"the price of protection that turned out to be unnecessary.")
        else:
            lines.append(
                f"With no move at all the hedge is {money(flat_cost)} ahead, "
                f"which happens only when the option was bought below its "
                f"model value.")

    if largest_benefit is not None and largest_shock is not None:
        lines.append(
            f"The largest benefit, {money(largest_benefit)}, comes at a "
            f"{largest_shock:+.0%} shock. These scenarios carry no probability "
            f"— that is the point of a stress test, and why it does not depend "
            f"on whether the model believes such a fall is likely.")

    return _drop_empty(lines)


def numerical_methods(convergence_order=None, theoretical=-0.5,
                      interpolation_error=None, grid_nodes=None):
    """Reading the convergence and interpolation evidence."""
    lines = []

    if convergence_order is not None:
        lines.append(
            f"Fitting the error against the number of paths gives an order of "
            f"{convergence_order:.3f}, against the {theoretical:.1f} that "
            f"Monte Carlo theory predicts. Matching that rate is how we know "
            f"the estimator is behaving, not merely producing a number.")

    if interpolation_error is not None and grid_nodes is not None:
        lines.append(
            f"The option is priced at {grid_nodes} spot values once and read "
            f"back by interpolation for every scenario, with a worst error of "
            f"{money(interpolation_error, 4)} per share. Pricing each scenario "
            f"directly would be a nested Monte Carlo and would take days.")

    return _drop_empty(lines)


def volatility_note(historical=None, implied=None):
    """Why two different volatilities appear on the same page."""
    if historical is None or implied is None:
        return []
    higher, lower = ("implied", "historical") if implied > historical \
        else ("historical", "implied")
    return [
        f"Two volatilities appear here and they mean different things. "
        f"{volatility(historical)} is what SPY actually realised over the "
        f"cached history; {volatility(implied)} is what the option's price "
        f"implies the market expects. The {higher} figure is the larger one, "
        f"and the gap between them is a view about the future, not an error."
    ]


def project_summary(early_premium=None, relative_error=None, heldout_mae=None,
                    n_heldout=None, cvar_reduction=None, n_candidates=None,
                    bootstrap_gap=None):
    """The project in one sentence, assembled from what was actually computed.

    Written as a template with the numbers filled in rather than a fixed
    sentence, so it cannot survive the results changing underneath it. Clauses
    whose phase has not been run are dropped instead of being invented.
    """
    clauses = ["We extend European Monte Carlo option pricing to American "
               "early exercise"]

    if early_premium is not None:
        clauses[-1] += f" (worth {price(early_premium)} of this contract)"

    if relative_error is not None:
        clauses.append(f"check it against an independent lattice to "
                       f"{percent(relative_error)}")

    if heldout_mae is not None and n_heldout is not None:
        clauses.append(f"test it on {int(n_heldout)} held-out market contracts "
                       f"to {money(heldout_mae, 4)}")

    if cvar_reduction is not None:
        clauses.append(f"use it to hedge a real SPY position, cutting 99% CVaR "
                       f"by {percent(cvar_reduction)}")

    if bootstrap_gap is not None:
        clauses.append(f"confirm that holds when risk is resampled from "
                       f"observed returns instead of assumed normal "
                       f"({percent(bootstrap_gap)} apart)")

    if n_candidates is not None:
        clauses.append(f"and search {int(n_candidates)} real listed strikes for "
                       f"the best protection per dollar spent")

    if len(clauses) == 1:
        return clauses[0] + "."
    return ", ".join(clauses[:-1]) + ", " + clauses[-1] + "."
