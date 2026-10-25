"""Shared fixtures and paths for the test suite."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

# GUI tests and benchmarks render offscreen unless a platform is chosen explicitly.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"
REFERENCE_DIR = TESTS_DIR / "reference"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic_data_dir() -> Path:
    return DATA_DIR / "synthetic"


@pytest.fixture(scope="session")
def synthetic_reference_dir() -> Path:
    return REFERENCE_DIR / "synthetic"


@pytest.fixture(scope="session")
def tolerances() -> dict[str, Any]:
    """Parsed ``tests/tolerances.yaml``."""
    with (TESTS_DIR / "tolerances.yaml").open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


LFS_POINTER_PREFIX = b"version https://git-lfs"


def _real_data_file(name: str) -> Path:
    path = DATA_DIR / "real" / name
    if not path.is_file():
        pytest.skip(f"real data file {name} is not present")
    with path.open("rb") as fh:
        if fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            pytest.skip(f"real data file {name} is an unfetched Git LFS pointer")
    return path


@pytest.fixture(scope="session")
def real_data_file() -> Callable[[str], Path]:
    """Return a function giving the path of a file in ``tests/data/real``.

    The function skips the calling test when the file is absent, including when it is a
    Git LFS pointer that was never pulled.
    """
    return _real_data_file


@pytest.fixture(scope="session")
def qapp() -> Iterator[Any]:
    """The session's QApplication; skips when PySide6 or pyqtgraph is not installed."""
    widgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from ephys_forever.gui.app import configure_pyqtgraph

    app = widgets.QApplication.instance() or widgets.QApplication([])
    configure_pyqtgraph()
    yield app


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run tests/benchmarks only under ``pytest --benchmark-only``, as CLAUDE.md lists."""
    if config.getoption("benchmark_only", default=False):
        return
    skip = pytest.mark.skip(reason="benchmark: run with pytest --benchmark-only")
    bench_dir = TESTS_DIR / "benchmarks"
    for item in items:
        if bench_dir in Path(str(item.fspath)).parents:
            item.add_marker(skip)
