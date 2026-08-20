"""VaR and CVaR, including the Rockafellar-Uryasev characterisation."""
import numpy as np
import pytest
from scipy.stats import norm

from src.portfolio import protective_put_portfolio, unhedged_portfolio
from src.sanity import check_risk_measures
from src.var_cvar import (bootstrap_risk_measures, cvar_by_minimisation,
                          conditional_value_at_risk, risk_measures,
                          risk_reduction, risk_table,
                          rockafellar_uryasev_objective, value_at_risk)


def test_var_is_the_quantile_of_losses():
    losses = np.arange(1.0, 101.0)          # 1 .. 100
    assert value_at_risk(losses, 0.95) == pytest.approx(np.quantile(losses, 0.95))
    assert value_at_risk(losses, 0.50) == pytest.approx(np.median(losses))


def test_cvar_is_the_mean_of_the_tail():
    losses = np.arange(1.0, 101.0)
    var = value_at_risk(losses, 0.95)
    expected = losses[losses >= var].mean()
    assert conditional_value_at_risk(losses, 0.95) == pytest.approx(expected)


def test_cvar_never_falls_below_var():
    """The property Rockafellar & Uryasev rely on: CVaR >= VaR always."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        losses = rng.standard_t(df=3, size=5_000) * rng.uniform(1, 100)
        for level in (0.90, 0.95, 0.99):
            var = value_at_risk(losses, level)
            cvar = conditional_value_at_risk(losses, level)
            assert cvar >= var - 1e-9
            assert check_risk_measures(var, cvar, level).passed


def test_matches_the_closed_form_for_a_normal_loss():
    """For L ~ N(mu, sigma): VaR = mu + sigma z, CVaR = mu + sigma phi(z)/(1-b)."""
    mu, sigma, level = 500.0, 2_000.0, 0.99
    rng = np.random.default_rng(11)
    losses = rng.normal(mu, sigma, 400_000)

    z = norm.ppf(level)
    expected_var = mu + sigma * z
    expected_cvar = mu + sigma * norm.pdf(z) / (1.0 - level)

    assert value_at_risk(losses, level) == pytest.approx(expected_var, rel=0.01)
    assert conditional_value_at_risk(losses, level) == pytest.approx(
        expected_cvar, rel=0.01)


def test_rockafellar_uryasev_theorem_one():
    """min_alpha F_beta(alpha) is the CVaR, and the minimiser is the VaR."""
    rng = np.random.default_rng(5)
    losses = rng.normal(300.0, 1_500.0, 200_000)

    for level in (0.95, 0.99):
        result = cvar_by_minimisation(losses, level)
        assert result["cvar"] == pytest.approx(
            conditional_value_at_risk(losses, level), rel=1e-4)
        assert result["alpha_star"] == pytest.approx(
            result["empirical_var"], rel=1e-3)


def test_objective_is_minimised_at_the_var_and_not_before():
    """F_beta must be convex with its trough at the VaR."""
    rng = np.random.default_rng(9)
    losses = rng.normal(0.0, 1_000.0, 100_000)
    level = 0.95
    var = value_at_risk(losses, level)

    at_var = rockafellar_uryasev_objective(losses, var, level)
    for offset in (-500.0, -100.0, 100.0, 500.0):
        assert rockafellar_uryasev_objective(losses, var + offset, level) >= at_var


def test_objective_at_the_var_equals_the_cvar():
    """Theorem 1 again, evaluated directly rather than by search."""
    rng = np.random.default_rng(13)
    losses = rng.normal(100.0, 800.0, 200_000)
    level = 0.95
    var = value_at_risk(losses, level)
    assert rockafellar_uryasev_objective(losses, var, level) == pytest.approx(
        conditional_value_at_risk(losses, level), rel=1e-3)


def test_degenerate_losses_are_handled():
    constant = np.full(100, 42.0)
    assert value_at_risk(constant, 0.95) == pytest.approx(42.0)
    assert conditional_value_at_risk(constant, 0.95) == pytest.approx(42.0)
    result = cvar_by_minimisation(constant, 0.95)
    assert result["cvar"] == pytest.approx(42.0)


def test_risk_measures_reports_dollars_and_percentages():
    losses = np.arange(1.0, 1001.0)
    measures = risk_measures(losses, (0.95, 0.99), initial_value=10_000.0)

    assert set(measures) == {"var_95", "cvar_95", "var_95_pct", "cvar_95_pct",
                             "var_99", "cvar_99", "var_99_pct", "cvar_99_pct"}
    assert measures["var_95_pct"] == pytest.approx(measures["var_95"] / 10_000.0)
    assert measures["cvar_99"] >= measures["var_99"]


def test_risk_reduction_sign_is_preserved():
    """A hedge that makes things worse must report a negative number."""
    assert risk_reduction(100.0, 40.0) == pytest.approx(60.0)
    assert risk_reduction(100.0, 100.0) == pytest.approx(0.0)
    assert risk_reduction(100.0, 150.0) == pytest.approx(-50.0)
    assert np.isnan(risk_reduction(0.0, 10.0))


def test_risk_table_has_a_reduction_row():
    spots = np.linspace(600.0, 900.0, 2_000)
    puts = np.maximum(749.0 - spots, 0.0) + 2.0
    a = unhedged_portfolio(100, 768.37, spots)
    b = protective_put_portfolio(100, 768.37, spots, 13.53, puts)

    table = risk_table([a, b], (0.95, 0.99))
    assert list(table["portfolio"]) == ["SPY only", "SPY + put", "risk reduction %"]

    baseline, hedged, reduction = table.iloc[0], table.iloc[1], table.iloc[2]
    assert reduction["var_95"] == pytest.approx(
        risk_reduction(baseline["var_95"], hedged["var_95"]))
    assert reduction["cvar_99"] == pytest.approx(
        risk_reduction(baseline["cvar_99"], hedged["cvar_99"]))


def test_percentage_table_uses_each_portfolio_own_base():
    spots = np.linspace(600.0, 900.0, 2_000)
    puts = np.maximum(749.0 - spots, 0.0) + 2.0
    a = unhedged_portfolio(100, 768.37, spots)
    b = protective_put_portfolio(100, 768.37, spots, 13.53, puts)

    dollars = risk_table([a, b], (0.95,))
    percent = risk_table([a, b], (0.95,), use_percentage=True)

    assert percent.iloc[0]["var_95"] == pytest.approx(
        dollars.iloc[0]["var_95"] / a.initial_value)
    assert percent.iloc[1]["var_95"] == pytest.approx(
        dollars.iloc[1]["var_95"] / b.initial_value)


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    losses = rng.normal(500.0, 2_000.0, 50_000)
    boot = bootstrap_risk_measures(losses, (0.95, 0.99), n_bootstrap=200,
                                   seed=17)

    for level in (0.95, 0.99):
        key = f"{level:.0%}".replace("%", "")
        var = value_at_risk(losses, level)
        cvar = conditional_value_at_risk(losses, level)
        assert boot[f"var_{key}_ci_low"] <= var <= boot[f"var_{key}_ci_high"]
        assert boot[f"cvar_{key}_ci_low"] <= cvar <= boot[f"cvar_{key}_ci_high"]
        assert boot[f"var_{key}_std_error"] > 0


def test_deeper_tails_are_estimated_less_precisely():
    """99% CVaR rests on a tenth as much data as 95%, so it must be noisier."""
    rng = np.random.default_rng(4)
    losses = rng.normal(0.0, 1_000.0, 50_000)
    boot = bootstrap_risk_measures(losses, (0.95, 0.99), n_bootstrap=300,
                                   seed=21)
    assert boot["cvar_99_std_error"] > boot["cvar_95_std_error"]


def test_rejects_bad_inputs():
    losses = np.arange(10.0)
    for level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            value_at_risk(losses, level)
    with pytest.raises(ValueError):
        conditional_value_at_risk(np.array([]), 0.95)
    with pytest.raises(ValueError):
        value_at_risk(np.zeros((3, 3)), 0.95)
