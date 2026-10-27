"""Curve fitting: models and least-squares fits of traces and tables."""

from ephys_forever.core.analysis.fit.core import FitError, FitResult, curve, fit_curve, fit_region
from ephys_forever.core.analysis.fit.expression import Expression, ExpressionError
from ephys_forever.core.analysis.fit.models import (
    BUILT_IN,
    Model,
    boltzmann,
    custom,
    exponential,
    hill,
)

__all__ = [
    "BUILT_IN",
    "Expression",
    "ExpressionError",
    "FitError",
    "FitResult",
    "Model",
    "boltzmann",
    "curve",
    "custom",
    "exponential",
    "fit_curve",
    "fit_region",
    "hill",
]
