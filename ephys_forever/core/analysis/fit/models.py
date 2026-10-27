"""Fit models: exponentials (1 to 3 terms), Boltzmann, Hill, and custom expressions.

A :class:`Model` bundles a function ``f(x, p)``, its parameter names, bounds, and a
data-driven start-value guess, so :func:`~ephys_forever.core.analysis.fit.fit_curve`
needs no start values from the caller for the built-in models.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ephys_forever.core.analysis.fit.expression import Expression
from ephys_forever.core.model import FloatArray

GUESS_POINTS = 5000
"""At most this many evenly spaced points are used to compute start values."""


@dataclass(frozen=True)
class Model:
    """A curve ``y = f(x, p)`` with named parameters.

    Attributes
    ----------
    name : str
        Short name recorded with results, for example ``"exponential_2"``.
    formula : str
        Human-readable formula.
    parameters : tuple of str
        Parameter names in the order ``f`` takes them.
    function : callable
        ``f(x, p)`` with ``p`` a 1-D array in :attr:`parameters` order.
    guess : callable or None
        ``guess(x, y) -> p0``; ``None`` means start values must be given.
    lower, upper : tuple of float
        Bounds for each parameter (``-inf``/``inf`` when unbounded).
    canonical : callable
        Reorders a fitted parameter vector into a standard form (exponential terms by
        increasing tau); identity for other models.
    """

    name: str
    formula: str
    parameters: tuple[str, ...]
    function: Callable[[FloatArray, FloatArray], FloatArray]
    guess: Callable[[FloatArray, FloatArray], FloatArray] | None = None
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()
    canonical: Callable[[FloatArray], FloatArray] = field(default=lambda p: p)

    def __post_init__(self) -> None:
        n = len(self.parameters)
        if not self.lower:
            object.__setattr__(self, "lower", (-math.inf,) * n)
        if not self.upper:
            object.__setattr__(self, "upper", (math.inf,) * n)
        if len(self.lower) != n or len(self.upper) != n:
            raise ValueError(f"{self.name}: bounds must have {n} entries")

    def __call__(self, x: FloatArray, params: Mapping[str, float] | Sequence[float]) -> FloatArray:
        """Evaluate with parameters by name or in order."""
        if isinstance(params, Mapping):
            p = np.array([params[n] for n in self.parameters], dtype=np.float64)
        else:
            p = np.asarray(params, dtype=np.float64)
        return self.function(np.asarray(x, dtype=np.float64), p)


def _thin(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    step = max(1, x.size // GUESS_POINTS)
    return x[::step], y[::step]


# -- exponentials -----------------------------------------------------------------
GRID_SIZE = {1: 40, 2: 24, 3: 14}
"""Candidate time constants per term count for the start-value search."""

RISING_MIN_SPANS = 20.0
"""Rising (negative) start time constants are at least the fitted span divided by this."""


def _exp_function(n_terms: int) -> Callable[[FloatArray, FloatArray], FloatArray]:
    def f(x: FloatArray, p: FloatArray) -> FloatArray:
        y = np.full(x.shape, p[-1], dtype=np.float64)
        for i in range(n_terms):
            y += p[2 * i] * np.exp(-x / p[2 * i + 1])
        return y

    return f


def _exp_guess(n_terms: int) -> Callable[[FloatArray, FloatArray], FloatArray]:
    def guess(x: FloatArray, y: FloatArray) -> FloatArray:
        xs, ys = _thin(x - x[0], y)
        span = float(xs[-1] - xs[0])
        dt = float(np.min(np.diff(xs))) if xs.size > 1 else span
        if span <= 0:
            raise ValueError("an exponential fit needs at least two distinct x values")
        decaying = np.geomspace(max(dt, span * 1e-4) / 2, span * 3, GRID_SIZE[n_terms])
        # Rising candidates stop at |tau| = span / RISING_MIN_SPANS so exp(-x / tau)
        # stays well inside float range over the fitted span.
        rising = -np.geomspace(span / RISING_MIN_SPANS, span * 3, GRID_SIZE[n_terms] // 2)
        taus = np.concatenate([decaying, rising])
        best: tuple[float, FloatArray] | None = None
        for combo in itertools.combinations(taus, n_terms):
            design = np.column_stack([*(np.exp(-xs / t) for t in combo), np.ones_like(xs)])
            coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
            sse = float(np.sum((design @ coef - ys) ** 2))
            if best is None or sse < best[0]:
                p = np.empty(2 * n_terms + 1)
                p[0:-1:2] = coef[:-1]
                p[1:-1:2] = combo
                p[-1] = coef[-1]
                best = (sse, p)
        assert best is not None
        p = best[1]
        # Amplitudes refer to x measured from x[0]; shift them to x = 0.
        p[0:-1:2] *= np.exp(x[0] / p[1:-1:2])
        return p

    return guess


def _exp_canonical(p: FloatArray) -> FloatArray:
    pairs = sorted(zip(p[0:-1:2], p[1:-1:2], strict=True), key=lambda ab: ab[1])
    out = np.array([v for pair in pairs for v in pair] + [p[-1]], dtype=np.float64)
    return out


def exponential(n_terms: int = 1) -> Model:
    """Sum of decaying exponentials plus a constant.

    ``y = A1 exp(-x / tau1) + ... + An exp(-x / taun) + C``, ``n_terms`` from 1 to 3.
    ``x`` is the time from the start of the fitted region (see
    :func:`~ephys_forever.core.analysis.fit.fit_region`), so ``A`` values are amplitudes
    at the first fitted sample. Time constants are not bounded: a negative ``tau`` is
    a rising exponential, as the commercial reference tool also reports. Terms are
    reported in increasing order of ``tau``.

    Start values come from a grid search: for every combination of candidate time
    constants (log-spaced from half the sample interval to three times the fitted
    span, plus rising ones from -span / 20 to -3 span) the amplitudes and offset are
    solved linearly, and the combination with the smallest squared error starts the
    nonlinear fit. This makes the fit independent of user-supplied start values.
    """
    if n_terms not in GRID_SIZE:
        raise ValueError(f"exponential fits have 1 to 3 terms, got {n_terms}")
    names = [n for i in range(1, n_terms + 1) for n in (f"A{i}", f"tau{i}")] + ["C"]
    terms = " + ".join(f"A{i} exp(-x / tau{i})" for i in range(1, n_terms + 1))
    return Model(
        name=f"exponential_{n_terms}",
        formula=f"{terms} + C",
        parameters=tuple(names),
        function=_exp_function(n_terms),
        guess=_exp_guess(n_terms),
        canonical=_exp_canonical,
    )


# -- Boltzmann --------------------------------------------------------------------
def _sorted_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _half_point(x: FloatArray, y: FloatArray, level: float) -> float:
    """x where y is closest to ``level``."""
    return float(x[int(np.argmin(np.abs(y - level)))])


def _boltzmann_function(x: FloatArray, p: FloatArray) -> FloatArray:
    top, v_half, k = p
    with np.errstate(over="ignore"):
        return np.asarray(top / (1.0 + np.exp((v_half - x) / k)), dtype=np.float64)


def _boltzmann_guess(x: FloatArray, y: FloatArray) -> FloatArray:
    xs, ys = _sorted_xy(x, y)
    rising = np.mean(ys[-max(1, ys.size // 4) :]) >= np.mean(ys[: max(1, ys.size // 4)])
    top = float(ys.max() if abs(ys.max()) >= abs(ys.min()) else ys.min())
    v_half = _half_point(xs, ys, top / 2)
    k = float(np.ptp(xs)) / 10 or 1.0
    return np.array([top, v_half, k if rising == (top > 0) else -k])


def boltzmann() -> Model:
    """Boltzmann ``y = top / (1 + exp((V_half - x) / k))``.

    ``k > 0`` describes activation (``y`` rises with ``x``) and ``k < 0`` inactivation.
    Used for conductance-voltage curves (``x`` in V, ``top`` = Gmax) and normalised
    availability curves.
    """
    return Model(
        name="boltzmann",
        formula="top / (1 + exp((V_half - x) / k))",
        parameters=("top", "V_half", "k"),
        function=_boltzmann_function,
        guess=_boltzmann_guess,
    )


# -- Hill -------------------------------------------------------------------------
def _hill_function(x: FloatArray, p: FloatArray) -> FloatArray:
    bottom, top, ec50, n = p
    with np.errstate(all="ignore"):
        xn = np.power(np.abs(x), n)
        return np.asarray(bottom + (top - bottom) * xn / (np.abs(ec50) ** n + xn), dtype=np.float64)


def _hill_guess(x: FloatArray, y: FloatArray) -> FloatArray:
    xs, ys = _sorted_xy(x, y)
    positive = xs > 0
    if not positive.any():
        raise ValueError("a Hill fit needs positive x values (concentrations)")
    quarter = max(1, ys.size // 4)
    bottom, top = float(np.mean(ys[:quarter])), float(np.mean(ys[-quarter:]))
    ec50 = _half_point(xs[positive], ys[positive], (bottom + top) / 2)
    return np.array([bottom, top, ec50, 1.0])


def hill() -> Model:
    """Hill equation ``y = bottom + (top - bottom) x^n / (EC50^n + x^n)``.

    ``x`` must be positive (a concentration); ``EC50`` is bounded positive and ``n`` is
    the Hill coefficient. A negative ``n`` describes inhibition (use ``IC50`` reading of
    ``EC50``).
    """
    return Model(
        name="hill",
        formula="bottom + (top - bottom) x^n / (EC50^n + x^n)",
        parameters=("bottom", "top", "EC50", "n"),
        function=_hill_function,
        guess=_hill_guess,
        lower=(-math.inf, -math.inf, 1e-300, -math.inf),
    )


# -- custom -----------------------------------------------------------------------
def custom(expression: str) -> Model:
    """Model from a typed expression in ``x``, for example ``"A * exp(-x / tau) + C"``.

    Parameter names are taken from the expression in order of first use. Custom models
    have no automatic start values: pass ``p0`` to the fit. See
    :class:`~ephys_forever.core.analysis.fit.expression.Expression` for what an
    expression may contain.
    """
    expr = Expression(expression)
    return Model(
        name="custom",
        formula=expression,
        parameters=expr.parameters,
        function=lambda x, p: expr(x, p),
    )


BUILT_IN: dict[str, Callable[[], Model]] = {
    "exponential_1": lambda: exponential(1),
    "exponential_2": lambda: exponential(2),
    "exponential_3": lambda: exponential(3),
    "boltzmann": boltzmann,
    "hill": hill,
}
"""Built-in models by name."""
