"""Tests for the presentation layer's pure functions.

Formatting and wording are where a dashboard quietly lies: a nan rendered as
"nan", a fraction shown as if it were a percentage, or an explanation that
says "the two methods agree closely" no matter what the numbers were. None of
these raise, so none of them are caught by running the app.
"""
import math

import pytest

from ui import explanations
from ui.formatters import (MISSING, count, days, fraction_as_percent, money,
                           moneyness, percent, price, ratio, signed_money,
                           signed_percent, snapshot_caption, strike_label,
                           volatility)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def test_money_uses_thousands_separators_and_two_decimals():
    assert money(1234.5) == "$1,234.50"
    assert money(76836.999512) == "$76,837.00"
    assert money(0.0033, 4) == "$0.0033"


def test_signed_money_always_shows_the_direction():
    assert signed_money(1602.57) == "+$1,602.57"
    assert signed_money(-200.98) == "-$200.98"


def test_percentage_helpers_do_not_confuse_their_units():
    """55.12 means 55.12%; 0.5512 also means 55.12%. Mixing them is the bug."""
    assert percent(55.1193) == "55.12%"
    assert fraction_as_percent(0.551193) == "55.12%"
    assert volatility(0.172149) == "17.21%"
    assert signed_percent(-6.2) == "-6.20%"


def test_price_keeps_four_decimals():
    assert price(13.511056) == "13.5111"


def test_small_helpers():
    assert ratio(1.87) == "1.87x"
    assert count(50000) == "50,000"
    assert days(73) == "73 days"
    assert strike_label(749.0) == "K=749"
    assert moneyness(0.9748) == "97.5% of spot"


@pytest.mark.parametrize("formatter", [money, signed_money, percent,
                                       fraction_as_percent, price, ratio,
                                       count, volatility, strike_label,
                                       moneyness, days, signed_percent])
def test_every_formatter_renders_missing_values_as_a_dash(formatter):
    """A metric card printing 'nan' tells the viewer nothing and looks broken."""
    assert formatter(None) == MISSING
    assert formatter(float("nan")) == MISSING


def test_infinite_values_are_also_treated_as_missing():
    assert money(math.inf) == MISSING
    assert ratio(-math.inf) == MISSING


def test_the_snapshot_caption_reads_as_a_date():
    caption = snapshot_caption("2026-08-18", "SPY", 73)
    assert "18 Aug 2026" in caption
    assert "SPY" in caption
    assert "73 DTE" in caption


def test_an_unparseable_date_is_passed_through_rather_than_crashing():
    assert "not-a-date" in snapshot_caption("not-a-date", "SPY", 10)


# --------------------------------------------------------------------------
# Explanations must be able to say bad news
# --------------------------------------------------------------------------

def test_pricing_explanation_calls_close_agreement_close():
    lines = " ".join(explanations.pricing(lsmc=13.5111, binomial=13.5799,
                                          std_error=0.2128))
    assert "agree" in lines
    assert "13.5111" in lines and "13.5799" in lines


def test_pricing_explanation_does_not_praise_a_bad_disagreement():
    """The sentence has to change when the numbers are poor, or it is decor."""
    good = " ".join(explanations.pricing(lsmc=13.51, binomial=13.58))
    bad = " ".join(explanations.pricing(lsmc=13.51, binomial=20.00))
    assert "agree" in good
    assert "agree" not in bad
    assert "too far apart" in bad


def test_hedging_explanation_reports_a_hedge_that_did_not_work():
    helped = " ".join(explanations.hedging(cvar_reduction=55.12))
    hurt = " ".join(explanations.hedging(cvar_reduction=-3.0))
    assert "removes" in helped
    assert "did not reduce" in hurt


def test_hedging_explanation_distinguishes_good_and_bad_value():
    rich = " ".join(explanations.hedging(saved_per_dollar=2.59))
    poor = " ".join(explanations.hedging(saved_per_dollar=0.4))
    assert "more than" in rich
    assert "less than" in poor


def test_risk_measures_explanation_separates_var_from_cvar():
    lines = " ".join(explanations.risk_measures(var=5598.41, cvar=6453.24,
                                                level=0.99))
    assert "where the bad tail begins" in lines
    assert "how bad is it on average" in lines
    assert "$854.83" in lines          # the gap between them


def test_risk_models_explanation_names_whichever_model_is_harsher():
    boot_harsher = " ".join(explanations.risk_models(gbm_cvar=6453.24,
                                                     bootstrap_cvar=6852.33))
    gbm_harsher = " ".join(explanations.risk_models(gbm_cvar=7000.0,
                                                    bootstrap_cvar=6852.33))
    assert "comes from the historical bootstrap" in boot_harsher
    assert "comes from the GBM" in gbm_harsher


def test_validation_explanation_grades_the_fit_against_the_quote_size():
    close = " ".join(explanations.validation(mae=0.0188, mean_quote=14.82))
    loose = " ".join(explanations.validation(mae=3.0, mean_quote=14.82))
    assert "close fit" in close
    assert "loose fit" in loose


def test_stress_explanation_reports_the_cost_of_a_flat_market():
    lines = " ".join(explanations.stress(flat_cost=-200.98))
    # "costs -$200.98" would be a double negative; the verb carries the sign.
    assert "costs $200.98" in lines
    assert "-$200.98" not in lines
    assert "unnecessary" in lines


def test_volatility_note_names_the_larger_of_the_two():
    lines = " ".join(explanations.volatility_note(historical=0.172149,
                                                  implied=0.150628))
    assert "17.21%" in lines and "15.06%" in lines
    assert "historical figure is the larger" in lines


@pytest.mark.parametrize("generator", [
    explanations.pricing, explanations.hedging, explanations.validation,
    explanations.risk_measures, explanations.risk_models, explanations.stress,
    explanations.numerical_methods,
])
def test_every_generator_survives_having_nothing_to_say(generator):
    """A page whose phase has not run yet must render, not raise."""
    assert generator() == [] or isinstance(generator(), list)


def test_explanations_are_deterministic():
    """Same inputs, same words -- there is no model in the loop."""
    first = explanations.pricing(lsmc=13.5111, binomial=13.5799, market=12.25)
    second = explanations.pricing(lsmc=13.5111, binomial=13.5799, market=12.25)
    assert first == second


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

def test_pricing_params_are_hashable_so_the_cache_can_key_on_them():
    from ui.state import PricingParams

    params = PricingParams(spot=768.37, strike=749.0, time_to_expiry=0.2,
                           risk_free_rate=0.0379, dividend_yield=0.0121,
                           volatility=0.1721, n_paths=10_000, n_steps=50,
                           degree=2, seed=42)
    assert hash(params) == hash(params)
    assert params == PricingParams(**vars(params))
    with pytest.raises(AttributeError):
        params.spot = 1.0            # frozen: a cache key must not mutate


def test_the_defaults_match_the_pipeline_configuration():
    import config
    from ui import state

    assert state.DEFAULTS["n_paths"] == config.LSMC_N_PATHS
    assert state.DEFAULTS["seed"] == config.SEED
    assert state.DEFAULTS["ticker"] == config.TICKER
    assert state.DEFAULTS["data_mode"] == state.CACHED
