"""The sanity checks the scope requires before any result is reported.

Scope section 10. These are collected in one place so the same checks run in
the SPY pricing phase, in the experiments and in main.py, and so a failure is
reported as a failure rather than quietly ignored.

Nothing here asserts that the protective put reduces tail risk. That is a
result to be measured, not an assumption to be encoded (scope section 23).
"""
from dataclasses import dataclass


@dataclass
class Check:
    """One named check with its verdict and the numbers behind it."""
    name: str
    passed: bool
    detail: str

    def __str__(self):
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def check_american_put(american, european, S0, K, tolerance=1e-9):
    """Bounds every American put price must satisfy."""
    intrinsic = max(K - S0, 0.0)
    return [
        Check("American >= European",
              american >= european - tolerance,
              f"{american:.4f} vs {european:.4f}"),
        Check("American >= intrinsic value",
              american >= intrinsic - tolerance,
              f"{american:.4f} vs {intrinsic:.4f}"),
        Check("American <= strike",
              american <= K + tolerance,
              f"{american:.4f} vs {K:.4f}"),
    ]


def check_convergence(values, label, tolerance):
    """The last step of a refinement sequence must move less than `tolerance`.

    Used for "price stabilises as paths increase" and "benchmark stabilises as
    binomial steps increase".
    """
    if len(values) < 2:
        return Check(label, False, "need at least two values")
    change = abs(values[-1] - values[-2])
    return Check(label, change <= tolerance,
                 f"last change {change:.5f} <= {tolerance:.5f}")


def check_risk_measures(var, cvar, level, tolerance=1e-9):
    """CVaR is an average of the worst losses, so it can never be below VaR."""
    return Check(f"CVaR >= VaR at {level:.0%}",
                 cvar >= var - tolerance,
                 f"CVaR {cvar:,.2f} vs VaR {var:,.2f}")


def report(checks, printer=print):
    """Print every check and return True only if all of them passed."""
    for check in checks:
        printer(f"  {check}")
    failed = [c for c in checks if not c.passed]
    if failed:
        printer(f"  --> {len(failed)} of {len(checks)} checks FAILED")
    return not failed


def check_measure_separation(real_world_drift, risk_free_rate,
                             dividend_yield=0.0, tolerance=1e-9):
    """Guard against the project's single most dangerous mistake.

    Scope section 7: option pricing runs under the risk-neutral measure with
    drift r - q, while portfolio VaR runs under the real-world measure with the
    historical mu. Using r as the stock drift in the risk simulation would
    quietly understate the expected drift and hand back a wrong tail -- with no
    error message anywhere, because the code would run perfectly.

    So the two drifts are compared before the risk simulation starts. They are
    estimated in completely different ways and would only coincide by accident.
    """
    risk_neutral = risk_free_rate - dividend_yield
    separated = (abs(real_world_drift - risk_neutral) > tolerance
                 and abs(real_world_drift - risk_free_rate) > tolerance)
    return Check(
        "risk simulation uses the real-world drift, not the risk-neutral one",
        separated,
        f"mu = {real_world_drift:.6f} vs r - q = {risk_neutral:.6f} "
        f"(r = {risk_free_rate:.6f})")
