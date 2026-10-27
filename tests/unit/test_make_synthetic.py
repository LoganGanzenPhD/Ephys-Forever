"""The synthetic generator is deterministic and the committed output is current."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_generator(repo_root: Path) -> ModuleType:
    path = repo_root / "scripts" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator(repo_root: Path) -> ModuleType:
    return load_generator(repo_root)


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_generator_is_byte_identical_across_runs(generator: ModuleType, tmp_path: Path) -> None:
    generator.generate(tmp_path / "a")
    generator.generate(tmp_path / "b")
    a, b = snapshot(tmp_path / "a"), snapshot(tmp_path / "b")
    assert len(a) == 2 * len(generator.CASES) + 1
    assert a == b


def test_every_case_listed_in_readme_is_generated(generator: ModuleType) -> None:
    names = {make.__name__ for make in generator.CASES}
    assert names >= {
        "step_known_amplitude",
        "exp_decay_single",
        "exp_decay_double",
        "boltzmann_activation",
        "epsc_train_known_times",
        "ap_train",
        "line_noise_60hz",
        "bessel_step_response",
        "exp_decay_triple",
        "hill_dose_response",
    }


def test_written_values_round_trip_within_format_precision(generator: ModuleType) -> None:
    case = generator.exp_decay_single()
    data = case.columns[1].data
    np.testing.assert_allclose(generator.as_written(data), data, rtol=1e-8, atol=0.0)


@pytest.mark.reference
def test_committed_synthetic_files_are_current(generator: ModuleType, repo_root: Path) -> None:
    assert generator.check(repo_root) == []
