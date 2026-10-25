"""SI unit handling and display conversion helpers.

All data inside ``ephys_forever.core`` is float64 in SI base units (s, A, V, S, Ω, F, Hz).
Readers use :func:`parse_unit` to turn a unit string from a file (for example ``"pA"``)
into an SI symbol and the factor that converts file values to SI. Display layers use
:func:`convert` to go back to prefixed units such as pA or mV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChannelKind = Literal["current", "voltage", "other"]

SI_BASE_UNITS: frozenset[str] = frozenset({"s", "A", "V", "S", "\u03a9", "F", "Hz"})
"""SI symbols the package stores data in."""

PREFIXES: dict[str, float] = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,  # micro sign U+00B5
    "μ": 1e-6,  # Greek mu U+03BC
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}
"""Decimal prefixes accepted in unit strings. ``u`` and both micro characters mean 1e-6."""

_ALIASES: dict[str, str] = {"Ohm": "\u03a9", "ohm": "\u03a9", "\u2126": "\u03a9"}
"""Alternative spellings of ohm, including the OHM SIGN character, mapped to U+03A9."""


class UnknownUnitError(ValueError):
    """Raised when a unit string is not a (prefixed) SI unit this package understands."""


@dataclass(frozen=True, slots=True)
class ParsedUnit:
    """A unit string split into its SI symbol and the factor to convert values to SI.

    Attributes
    ----------
    text : str
        The unit string as given.
    si_unit : str
        SI symbol, one of :data:`SI_BASE_UNITS`.
    factor : float
        Multiply a value in ``text`` units by this to get SI. ``pA`` gives 1e-12.
    """

    text: str
    si_unit: str
    factor: float


def parse_unit(text: str) -> ParsedUnit:
    """Parse a possibly prefixed unit string such as ``"pA"``, ``"mV"`` or ``"MOhm"``.

    Parameters
    ----------
    text : str
        Unit string. Surrounding whitespace is ignored.

    Returns
    -------
    ParsedUnit
        SI symbol and the factor from ``text`` units to SI.

    Raises
    ------
    UnknownUnitError
        If ``text`` is empty or is not a prefix from :data:`PREFIXES` followed by an SI
        symbol (or ``Ohm``). Callers decide how to handle such channels; nothing is
        converted silently.
    """
    stripped = text.strip()
    for base_text in sorted({*SI_BASE_UNITS, *_ALIASES}, key=len, reverse=True):
        if stripped.endswith(base_text):
            prefix = stripped[: len(stripped) - len(base_text)]
            if prefix in PREFIXES:
                si = _ALIASES.get(base_text, base_text)
                return ParsedUnit(text=text, si_unit=si, factor=PREFIXES[prefix])
    raise UnknownUnitError(
        f"unit {text!r} is not a prefixed SI unit; expected a prefix from "
        f"{sorted(p for p in PREFIXES if p)} followed by one of {sorted(SI_BASE_UNITS)}"
    )


def channel_kind(si_unit: str) -> ChannelKind:
    """Classify a channel from its SI unit: ``A`` is current, ``V`` is voltage."""
    if si_unit == "A":
        return "current"
    if si_unit == "V":
        return "voltage"
    return "other"


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units with the same SI base, e.g. ``"A"`` to ``"pA"``.

    Parameters
    ----------
    value : float
        Value in ``from_unit``.
    from_unit, to_unit : str
        Prefixed SI unit strings accepted by :func:`parse_unit`.

    Returns
    -------
    float
        ``value`` expressed in ``to_unit``.

    Raises
    ------
    UnknownUnitError
        If either unit cannot be parsed.
    ValueError
        If the units have different SI bases, for example ``"mV"`` and ``"pA"``.
    """
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src.si_unit != dst.si_unit:
        raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}: different quantities")
    if src.factor == dst.factor:
        return value
    return value * src.factor / dst.factor
