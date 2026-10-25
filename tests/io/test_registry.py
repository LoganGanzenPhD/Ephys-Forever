"""Format detection and dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from ephys_forever.core.io.errors import ReaderError, UnsupportedFeatureError
from ephys_forever.core.io.registry import detect_format, read

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.mark.parametrize(
    ("head", "fmt"), [(b"ABF2", "abf2"), (b"ABF ", "abf1"), (b"ATF\t1.0", "atf")]
)
def test_detect_format(tmp_path: Path, head: bytes, fmt: str) -> None:
    path = tmp_path / "f"
    path.write_bytes(head + bytes(16))
    assert detect_format(path) == fmt


def test_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "f.dat"
    path.write_bytes(b"HEKA")
    with pytest.raises(ReaderError, match="unrecognized"):
        read(path)


def test_abf1_not_supported_yet(tmp_path: Path) -> None:
    path = tmp_path / "old.abf"
    path.write_bytes(b"ABF " + bytes(16))
    with pytest.raises(UnsupportedFeatureError, match="abf1"):
        read(path)


def test_dispatches_to_atf() -> None:
    rec = read(SYNTHETIC / "exp_decay_single.atf")
    assert rec.metadata["format"] == "atf"
