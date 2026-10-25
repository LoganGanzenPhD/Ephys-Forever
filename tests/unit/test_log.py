"""Operation log: entries carry what a replay needs and survive a save and load."""

from __future__ import annotations

from pathlib import Path

import pytest

from ephys_forever import __version__
from ephys_forever.core.log import Operation, OperationLog


def test_record_save_and_load(tmp_path: Path) -> None:
    log = OperationLog()
    entry = log.record(
        "ephys_forever.core.analysis.measure.measure_sweeps",
        {"channel": 2, "region": [0.1, 0.2], "polarity": "negative"},
        input_sha256="ab" * 32,
        input_name="cell.abf",
    )
    assert entry.package_version == __version__
    assert entry.time.endswith("+00:00")
    path = tmp_path / "ops.json"
    log.save(path)
    back = OperationLog.load(path)
    assert back.entries == log.entries
    assert isinstance(back.entries[0], Operation)


def test_non_json_parameters_are_refused() -> None:
    with pytest.raises(TypeError):
        OperationLog().record("f", {"region": object()}, input_sha256="0" * 64)


def test_unknown_format_version_is_refused() -> None:
    with pytest.raises(ValueError, match="format version 99"):
        OperationLog.from_json('{"format_version": 99, "entries": []}')
