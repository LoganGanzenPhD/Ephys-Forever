"""Fit models, expressions, and fit argument checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ephys_forever.core.analysis.fit import (
    BUILT_IN,
    Expression,
    ExpressionError,
    FitError,
    boltzmann,
    curve,
    custom,
    exponential,
    fit_curve,
    hill,
)


def test_expression_parameters_and_evaluation() -> None:
    expr = Expression("A * exp(-x / tau) + C")
    assert expr.parameters == ("A", "tau", "C")
    x = np.array([0.0, 1.0])
    np.testing.assert_allclose(expr(x, {"A": 2.0, "tau": 1.0, "C": 1.0}), [3.0, 1 + 2 / math.e])
    np.testing.assert_allclose(Expression("a * x^2 + pi")(x, [3.0]), [math.pi, 3 + math.pi])


def test_expression_constant_in_x_still_broadcasts() -> None:
    assert Expression("0 * x + c")(np.zeros(4), [2.0]).tolist() == [2.0] * 4


def test_expression_division_by_zero_gives_inf_not_an_exception() -> None:
    out = Expression("a / x")(np.array([0.0, 2.0]), [1.0])
    assert math.isinf(out[0]) and out[1] == 0.5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("A * exp(", "cannot parse"),
        ("A + B", "does not use the variable"),
        ("__import__('os').system('x') + x", "only these functions"),
        ("x.real", "Attribute"),
        ("x if A else B", "IfExp"),
        ("exp + x", "must be called"),
        ("exp(x, 2)", "one argument"),
        ("'a' + x", "only numbers"),
        ("x % 2", "Mod"),
        ("[x][0]", "Subscript"),
    ],
)
def test_expression_rejects_anything_outside_the_whitelist(text: str, message: str) -> None:
    with pytest.raises(ExpressionError, match=message):
        Expression(text)


def test_exponential_terms_are_reported_by_increasing_tau() -> None:
    model = exponential(2)
    x = np.linspace(0, 0.1, 2000)
    y = model(x, [1.0, 0.03, 2.0, 0.002, 0.5])
    result = fit_curve(model, x, y)
    assert result.params["tau1"] == pytest.approx(0.002, rel=1e-6)
    assert result.params["A1"] == pytest.approx(2.0, rel=1e-6)
    assert result.params["tau2"] == pytest.approx(0.03, rel=1e-6)
    np.testing.assert_allclose(curve(model, result, x), y, rtol=1e-6)
    assert result.r_squared == pytest.approx(1.0)


def test_inactivation_boltzmann_has_negative_slope() -> None:
    v = np.linspace(-0.12, 0.0, 25)
    y = boltzmann()(v, [1.0, -0.06, -0.008])
    result = fit_curve(boltzmann(), v, y)
    assert result.params["k"] == pytest.approx(-0.008, rel=1e-6)
    assert result.params["V_half"] == pytest.approx(-0.06, rel=1e-6)


def test_inhibition_hill_has_negative_coefficient() -> None:
    conc = np.logspace(-9, -4, 15)
    y = hill()(conc, [0.0, 1.0, 1e-6, -1.2])
    result = fit_curve(hill(), conc, y, [0.1, 0.9, 3e-6, -1.0])
    assert result.params["n"] == pytest.approx(-1.2, rel=1e-5)
    assert result.params["EC50"] == pytest.approx(1e-6, rel=1e-5)


def test_built_in_names() -> None:
    assert sorted(BUILT_IN) == [
        "boltzmann",
        "exponential_1",
        "exponential_2",
        "exponential_3",
        "hill",
    ]
    assert BUILT_IN["exponential_3"]().parameters == (
        "A1",
        "tau1",
        "A2",
        "tau2",
        "A3",
        "tau3",
        "C",
    )


def test_exact_fit_has_no_spare_degrees_of_freedom_error() -> None:
    with pytest.raises(FitError, match="only 3 points"):
        fit_curve(exponential(1), np.arange(3.0), np.ones(3))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((custom("a * x + b"), np.arange(5.0), np.arange(5.0)), "needs start values"),
        ((exponential(1), np.arange(5.0), np.arange(4.0)), "same length"),
        ((exponential(1), np.arange(5.0), np.r_[np.arange(4.0), np.nan]), "NaN"),
        ((hill(), -np.arange(1.0, 6.0), np.arange(5.0)), "positive x"),
    ],
)
def test_bad_fit_input_raises(args: tuple[object, ...], message: str) -> None:
    with pytest.raises((FitError, ValueError), match=message):
        fit_curve(*args)  # type: ignore[arg-type]


def test_partial_start_values_by_name_are_refused() -> None:
    with pytest.raises(FitError, match="missing for tau1, C"):
        fit_curve(exponential(1), np.arange(10.0), np.arange(10.0), {"A1": 1.0})


def test_exponential_terms_must_be_one_to_three() -> None:
    with pytest.raises(ValueError, match="1 to 3"):
        exponential(4)


def test_exponential_fits_a_rising_curve_with_negative_tau() -> None:
    x = np.linspace(0.0, 0.05, 1001)
    y = -3e-12 * np.exp(x / 0.02) + 1e-12  # tau = -20 ms
    result = fit_curve(exponential(1), x, y)
    assert result.params["tau1"] == pytest.approx(-0.02, rel=1e-6)
    assert result.params["A1"] == pytest.approx(-3e-12, rel=1e-6)
    assert result.params["C"] == pytest.approx(1e-12, rel=1e-6)
