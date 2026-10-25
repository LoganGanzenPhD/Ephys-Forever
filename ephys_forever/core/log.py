"""Operation log: a replayable record of every analysis run.

Each entry names the analysis function, every parameter it was called with, the
SHA-256 of the input file and the package version, so a later run can repeat the
analysis exactly. Parameters must be JSON values; time regions are stored as
``[start, stop]`` in seconds. Replaying a log headlessly is a later roadmap phase;
this module only records and stores it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

from ephys_forever import __version__

logger = logging.getLogger(__name__)

LOG_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Operation:
    """One analysis run.

    Attributes
    ----------
    function : str
        Dotted path of the analysis function, for example
        ``"ephys_forever.core.analysis.measure.measure_sweeps"``.
    params : dict
        Every argument other than the input data, as JSON values.
    input_sha256 : str
        SHA-256 of the input file's bytes, as readers store in ``metadata["sha256"]``.
    input_name : str
        File name of the input, for people reading the log; not used to match files.
    package_version : str
        Version of this package that ran the analysis.
    time : str
        UTC time of the run, ISO 8601.
    """

    function: str
    params: dict[str, Any]
    input_sha256: str
    input_name: str = ""
    package_version: str = __version__
    time: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class OperationLog:
    """Ordered list of :class:`Operation` entries that saves to and loads from JSON."""

    def __init__(self, entries: list[Operation] | None = None) -> None:
        self.entries: list[Operation] = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    def record(
        self,
        function: str,
        params: Mapping[str, Any],
        input_sha256: str,
        input_name: str = "",
    ) -> Operation:
        """Append an entry and return it.

        Raises
        ------
        TypeError
            If ``params`` holds a value that is not a JSON value.
        """
        params = dict(params)
        json.dumps(params)  # fail now, not when the log is saved
        entry = Operation(function, params, input_sha256, input_name)
        self.entries.append(entry)
        logger.info("recorded %s on %s", function, input_name or input_sha256[:12])
        return entry

    def to_json(self) -> str:
        return json.dumps(
            {"format_version": LOG_FORMAT_VERSION, "entries": [asdict(e) for e in self.entries]},
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> OperationLog:
        """Parse a log written by :meth:`to_json`.

        Raises
        ------
        ValueError
            If the text is not a log of a known format version.
        """
        data = json.loads(text)
        version = data.get("format_version") if isinstance(data, dict) else None
        if version != LOG_FORMAT_VERSION:
            raise ValueError(
                f"operation log format version {version!r} is not {LOG_FORMAT_VERSION}"
            )
        return cls([Operation(**e) for e in data["entries"]])

    def save(self, path: str | PathLike[str]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> OperationLog:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
