"""Format detection and dispatch: open any supported recording with :func:`read`."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path

from ephys_forever.core.io.abf2 import read_abf2
from ephys_forever.core.io.atf import read_atf
from ephys_forever.core.io.errors import ReaderError, UnsupportedFeatureError
from ephys_forever.core.model import Recording

Reader = Callable[[str | PathLike[str]], Recording]

FILE_FILTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Axon binary (*.abf)", (".abf",)),
    ("Axon text (*.atf)", (".atf",)),
)
"""Display name and extensions of every format :func:`read` opens."""


def detect_format(path: str | PathLike[str]) -> str:
    """Name the format of ``path`` from its first bytes: ``"abf2"``, ``"abf1"`` or ``"atf"``.

    Raises
    ------
    ReaderError
        If the leading bytes match no known format.
    """
    with Path(path).open("rb") as fh:
        head = fh.read(4)
    if head == b"ABF2":
        return "abf2"
    if head == b"ABF ":
        return "abf1"
    if head[:3] == b"ATF":
        return "atf"
    raise ReaderError(f"{Path(path).name}: unrecognized file format (starts with {head!r})")


READERS: dict[str, Reader] = {"abf2": read_abf2, "atf": read_atf}


def read(path: str | PathLike[str]) -> Recording:
    """Read a recording in any supported format.

    Raises
    ------
    ReaderError
        If the format is not recognized or the file is malformed.
    UnsupportedFeatureError
        If the format is recognized but has no reader yet (ABF 1.x).
    """
    fmt = detect_format(path)
    reader = READERS.get(fmt)
    if reader is None:
        raise UnsupportedFeatureError(f"{Path(path).name}: {fmt} files are not supported yet")
    return reader(path)
