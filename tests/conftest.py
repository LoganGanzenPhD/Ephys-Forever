"""Shared fixtures and paths for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

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
