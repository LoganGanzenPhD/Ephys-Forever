"""Safe evaluation of user-typed model expressions such as ``A * exp(-x / tau) + C``.

Expressions are parsed with :mod:`ast` and evaluated by walking the tree; nothing is
passed to ``eval``. Only numbers, the independent variable ``x``, named parameters,
arithmetic operators and the functions in :data:`FUNCTIONS` are allowed.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from ephys_forever.core.model import FloatArray

FUNCTIONS: dict[str, Callable[..., Any]] = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "arctan": np.arctan,
}
"""Functions an expression may call, all applied element-wise."""
CONSTANTS: dict[str, float] = {"pi": float(np.pi), "e": float(np.e)}
VARIABLE = "x"

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class ExpressionError(ValueError):
    """Raised for an expression that cannot be parsed or uses something not allowed."""


class Expression:
    """A parsed model expression in ``x`` and named parameters.

    Parameters
    ----------
    text : str
        The expression, for example ``"A * exp(-x / tau) + C"``. ``^`` is accepted as
        a power operator.

    Attributes
    ----------
    text : str
        The expression as given.
    parameters : tuple of str
        Names other than ``x``, functions and constants, in order of first use.

    Raises
    ------
    ExpressionError
        If the text is not a valid expression, or uses anything other than numbers,
        ``x``, parameter names, ``+ - * / **``, and :data:`FUNCTIONS`.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        try:
            tree = ast.parse(text.replace("^", "**"), mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"cannot parse {text!r}: {exc.msg}") from exc
        self._body = tree.body
        names: list[str] = []
        self._check(self._body, names)
        if VARIABLE not in names:
            raise ExpressionError(f"{text!r} does not use the variable {VARIABLE!r}")
        self.parameters = tuple(n for n in names if n != VARIABLE)

    def _check(self, node: ast.AST, names: list[str]) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ExpressionError(f"{self.text!r}: only numbers are allowed as constants")
        elif isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                raise ExpressionError(f"{self.text!r}: {node.id} must be called, as {node.id}(...)")
            if node.id not in CONSTANTS and node.id not in names:
                names.append(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ExpressionError(
                    f"{self.text!r}: operator {type(node.op).__name__} not allowed"
                )
            self._check(node.left, names)
            self._check(node.right, names)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise ExpressionError(
                    f"{self.text!r}: operator {type(node.op).__name__} not allowed"
                )
            self._check(node.operand, names)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                allowed = ", ".join(sorted(FUNCTIONS))
                raise ExpressionError(f"{self.text!r}: only these functions are allowed: {allowed}")
            if node.keywords or len(node.args) != 1:
                raise ExpressionError(f"{self.text!r}: {node.func.id} takes one argument")
            self._check(node.args[0], names)
        else:
            raise ExpressionError(f"{self.text!r}: {type(node).__name__} is not allowed")

    def __call__(
        self, x: FloatArray, params: Mapping[str, float] | Sequence[float] | FloatArray
    ) -> FloatArray:
        """Evaluate at ``x`` with parameter values by name or in :attr:`parameters` order."""
        if isinstance(params, Mapping):
            values = {n: float(params[n]) for n in self.parameters}
        else:
            if len(params) != len(self.parameters):
                raise ExpressionError(
                    f"{self.text!r} needs {len(self.parameters)} parameter values, "
                    f"got {len(params)}"
                )
            values = dict(zip(self.parameters, (float(p) for p in params), strict=True))
        env: dict[str, Any] = {k: np.float64(v) for k, v in {**CONSTANTS, **values}.items()}
        env[VARIABLE] = np.asarray(x, dtype=np.float64)
        with np.errstate(all="ignore"):
            result = self._eval(self._body, env)
        return np.broadcast_to(np.asarray(result, dtype=np.float64), np.shape(x)).copy()

    def _eval(self, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return np.float64(
                cast(float, node.value)
            )  # numpy scalars obey errstate; Python floats raise
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._eval(node.operand, env))
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)  # guaranteed by _check
            return FUNCTIONS[node.func.id](self._eval(node.args[0], env))
        raise ExpressionError(f"{type(node).__name__} is not allowed")  # unreachable
