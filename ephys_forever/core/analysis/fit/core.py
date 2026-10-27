"""Least-squares curve fitting and the fit result record."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ephys_forever.core.analysis.fit.models import Model
from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.model import FloatArray

FIT_TOLERANCE = 1e-12
"""``ftol``, ``xtol`` and ``gtol`` passed to :func:`scipy.optimize.least_squares`."""
MAX_EVALUATIONS = 20_000


class FitError(RuntimeError):
    """Raised when a fit cannot start (bad input) or the model cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of one least-squares fit.

    Attributes
    ----------
    model : str
        Model name.
    formula : str
        Model formula.
    params : dict of str to float
        Fitted values, SI units of ``x`` and ``y``.
    stderr : dict of str to float
        Asymptotic standard errors from the Jacobian at the solution, scaled by the
        residual variance. NaN when the fit has no spare degrees of freedom or the
        Jacobian is singular.
    sse : float
        Sum of squared residuals.
    residual_sd : float
        ``sqrt(sse / (n_points - n_params))``.
    r_squared : float
        ``1 - sse / sum((y - mean(y))^2)``.
    n_points : int
        Points fitted.
    x_offset : float
        Value subtracted from ``x`` before fitting (the region start for
        :func:`fit_region`, otherwise 0). Evaluate the curve with :meth:`curve`.
    converged : bool
        Whether the optimiser met a convergence criterion.
    message : str
        Optimiser status message.
    """

    model: str
    formula: str
    params: dict[str, float]
    stderr: dict[str, float]
    sse: float
    residual_sd: float
    r_squared: float
    n_points: int
    x_offset: float
    converged: bool
    message: str


def curve(model: Model, result: FitResult, x: FloatArray) -> FloatArray:
    """Fitted curve at ``x`` (same axis as the data passed to the fit)."""
    return model(np.asarray(x, dtype=np.float64) - result.x_offset, result.params)


def fit_curve(
    model: Model,
    x: FloatArray,
    y: FloatArray,
    p0: Mapping[str, float] | Sequence[float] | None = None,
    *,
    x_offset: float = 0.0,
) -> FitResult:
    """Fit ``model`` to ``(x, y)`` by unweighted least squares.

    Parameters
    ----------
    model : Model
        Model to fit, from :mod:`~ephys_forever.core.analysis.fit.models`.
    x, y : numpy.ndarray
        Data, 1-D, same length, SI units.
    p0 : mapping or sequence of float, optional
        Start values by name or in parameter order. Default: the model's own guess.
        Required for models without a guess (custom expressions).
    x_offset : float, optional
        Subtracted from ``x`` before fitting, so models see ``x - x_offset``.

    Returns
    -------
    FitResult

    Raises
    ------
    FitError
        If the data are unusable (too few points, non-finite values), start values are
        missing, or the model gives non-finite values at the start.

    Notes
    -----
    The optimiser is SciPy's trust-region reflective ``least_squares`` with bounds
    from the model. Residuals are divided by the spread of ``y`` and parameters are
    scaled by the Jacobian, so tiny SI values (picoamps, milliseconds) converge as well
    as values near one. A fit that hits the evaluation limit is returned with
    ``converged=False`` rather than raising.
    """
    xa = np.asarray(x, dtype=np.float64) - x_offset
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or xa.shape != ya.shape:
        raise FitError(f"x and y must be 1-D and the same length, got {xa.shape} and {ya.shape}")
    n_params = len(model.parameters)
    if xa.size <= n_params:
        raise FitError(f"{model.name} has {n_params} parameters but only {xa.size} points")
    if not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise FitError("data contain NaN or infinite values")

    if p0 is None:
        if model.guess is None:
            raise FitError(f"{model.name} needs start values for {', '.join(model.parameters)}")
        start = np.asarray(model.guess(xa, ya), dtype=np.float64)
    elif isinstance(p0, Mapping):
        missing = [n for n in model.parameters if n not in p0]
        if missing:
            raise FitError(f"start values missing for {', '.join(missing)}")
        start = np.array([p0[n] for n in model.parameters], dtype=np.float64)
    else:
        start = np.asarray(p0, dtype=np.float64)
        if start.shape != (n_params,):
            raise FitError(f"{model.name} needs {n_params} start values, got {start.size}")
    lower, upper = np.array(model.lower), np.array(model.upper)
    start = np.clip(start, lower, upper)
    if not np.isfinite(model.function(xa, start)).all():
        raise FitError(f"{model.name} is not finite at the start values {start.tolist()}")

    scale = float(np.std(ya)) or float(np.max(np.abs(ya))) or 1.0

    def residuals(p: FloatArray) -> FloatArray:
        return (model.function(xa, p) - ya) / scale

    solution = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=FIT_TOLERANCE,
        xtol=FIT_TOLERANCE,
        gtol=FIT_TOLERANCE,
        max_nfev=MAX_EVALUATIONS,
    )
    p = model.canonical(np.asarray(solution.x, dtype=np.float64))
    resid = model.function(xa, p) - ya
    sse = float(np.sum(resid**2))
    dof = xa.size - n_params
    sst = float(np.sum((ya - ya.mean()) ** 2))
    stderr = _standard_errors(model, xa, p, sse, dof)
    return FitResult(
        model=model.name,
        formula=model.formula,
        params=dict(zip(model.parameters, (float(v) for v in p), strict=True)),
        stderr=dict(zip(model.parameters, stderr, strict=True)),
        sse=sse,
        residual_sd=math.sqrt(sse / dof),
        r_squared=1.0 - sse / sst if sst > 0 else math.nan,
        n_points=int(xa.size),
        x_offset=float(x_offset),
        converged=bool(solution.status > 0),
        message=str(solution.message),
    )


def _standard_errors(
    model: Model, x: FloatArray, p: FloatArray, sse: float, dof: int
) -> list[float]:
    """sqrt(diag(inv(J^T J)) * sse / dof) with a central-difference Jacobian."""
    jac = np.empty((x.size, p.size))
    for j in range(p.size):
        h = 1e-6 * max(abs(p[j]), 1e-300)
        up, down = p.copy(), p.copy()
        up[j] += h
        down[j] -= h
        jac[:, j] = (model.function(x, up) - model.function(x, down)) / (2 * h)
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj) * (sse / dof)
    except np.linalg.LinAlgError:
        return [math.nan] * p.size
    diag = np.diag(cov)
    return [float(math.sqrt(v)) if v >= 0 and math.isfinite(v) else math.nan for v in diag]


def fit_region(
    model: Model,
    y: FloatArray,
    sample_rate: float,
    region: Region,
    p0: Mapping[str, float] | Sequence[float] | None = None,
) -> FitResult:
    """Fit ``model`` to one sweep's samples between two cursors, against time.

    Parameters
    ----------
    model : Model
        Model in time; ``x`` is seconds from the first sample in the region, so
        exponential amplitudes are values at that sample.
    y : numpy.ndarray
        One channel of one sweep, SI units.
    sample_rate : float
        Samples per second.
    region : Region
        Fit window; samples on both cursors are included.
    p0 : optional
        Start values, see :func:`fit_curve`.

    Returns
    -------
    FitResult
        ``x_offset`` is the time of the first fitted sample, so
        ``curve(model, result, t)`` takes sweep time ``t``.
    """
    i0, i1 = region.indices(sample_rate, np.asarray(y).size)
    t = np.arange(i0, i1, dtype=np.float64) / sample_rate
    return fit_curve(model, t, np.asarray(y)[i0:i1], p0, x_offset=i0 / sample_rate)
