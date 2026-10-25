"""SI unit parsing and display conversion."""

from __future__ import annotations

import pytest

from ephys_forever.core.units import UnknownUnitError, channel_kind, convert, parse_unit


@pytest.mark.parametrize(
    ("text", "si", "factor"),
    [
        ("A", "A", 1.0),
        ("pA", "A", 1e-12),
        ("nA", "A", 1e-9),
        ("mV", "V", 1e-3),
        ("V", "V", 1.0),
        ("uV", "V", 1e-6),
        ("µV", "V", 1e-6),
        ("μV", "V", 1e-6),
        ("ms", "s", 1e-3),
        ("s", "s", 1.0),
        ("nS", "S", 1e-9),
        ("pF", "F", 1e-12),
        ("kHz", "Hz", 1e3),
        ("MOhm", "Ω", 1e6),
        ("MΩ", "Ω", 1e6),
        ("GΩ", "Ω", 1e9),
        (" pA ", "A", 1e-12),
    ],
)
def test_parse_unit(text: str, si: str, factor: float) -> None:
    parsed = parse_unit(text)
    assert (parsed.si_unit, parsed.factor, parsed.text) == (si, factor, text)


@pytest.mark.parametrize("text", ["", "Degrees", "xA", "pa", "mm"])
def test_parse_unit_rejects_unknown(text: str) -> None:
    with pytest.raises(UnknownUnitError):
        parse_unit(text)


def test_channel_kind() -> None:
    assert channel_kind("A") == "current"
    assert channel_kind("V") == "voltage"
    assert channel_kind("S") == "other"


def test_convert() -> None:
    assert convert(-5e-11, "A", "pA") == pytest.approx(-50.0, rel=1e-15)
    assert convert(-60.0, "mV", "V") == pytest.approx(-0.06, rel=1e-15)
    assert convert(3.0, "mV", "mV") == 3.0


def test_convert_rejects_different_quantities() -> None:
    with pytest.raises(ValueError, match="different quantities"):
        convert(1.0, "mV", "pA")
