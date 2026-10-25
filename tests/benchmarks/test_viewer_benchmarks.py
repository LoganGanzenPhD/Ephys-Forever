"""Open, render and scroll timings (pytest-benchmark).

Run with ``pytest --benchmark-only``. The 30-minute gap-free cases check the CLAUDE.md
target of opening such a recording in under a second. Until a large real recording is
added, the file is built from the IV steps header switched to gap-free mode, with 30
minutes of random int16 samples on its 3 channels at 20 kHz (216 MB).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ephys_forever.core.io.registry import read
from ephys_forever.core.model import Channel, Recording, Sweep

GAP_FREE_SECONDS = 30 * 60
GAP_FREE_RATE = 20_000.0
TARGET_S = 1.0


@pytest.fixture
def iv_path(real_data_file: Callable[[str], Path]) -> Path:
    return real_data_file("vclamp_iv_steps_001.abf")


@pytest.fixture(scope="session")
def gap_free_abf(
    real_data_file: Callable[[str], Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """30 minutes x 3 channels at 20 kHz as ABF2 gap-free."""
    source = real_data_file("vclamp_iv_steps_001.abf").read_bytes()
    data_block = struct.unpack_from("<I", source, 76 + 16 * 10)[0]
    header = bytearray(source[: data_block * 512])
    total = int(GAP_FREE_SECONDS * GAP_FREE_RATE) * 3
    struct.pack_into("<h", header, 512, 3)  # nOperationMode = gap-free
    struct.pack_into("<I", header, 12, 1)  # uActualEpisodes
    struct.pack_into("<q", header, 76 + 16 * 10 + 8, total)  # Data section entries
    path = tmp_path_factory.mktemp("bench") / "gap_free_30min.abf"
    with path.open("wb") as fh:
        fh.write(header)
        np.random.default_rng(0).integers(-2000, 2000, total, dtype=np.int16).tofile(fh)
    return path


def render(qapp: Any, widget: Any) -> None:
    widget.repaint()
    qapp.processEvents()
    widget.grab()


@pytest.mark.realdata
def test_read_iv_abf(benchmark: Any, iv_path: Path) -> None:
    rec = benchmark(read, iv_path)
    assert rec.n_sweeps == 9


@pytest.mark.realdata
@pytest.mark.gui
def test_open_and_draw_iv(benchmark: Any, qapp: Any, iv_path: Path) -> None:
    from ephys_forever.gui.app import MainWindow

    window = MainWindow()
    window.show()

    def open_and_draw() -> None:
        window.open_path(iv_path)
        render(qapp, window)

    benchmark(open_and_draw)
    window.close()


@pytest.mark.realdata
@pytest.mark.gui
def test_step_through_sweeps(benchmark: Any, qapp: Any, iv_path: Path) -> None:
    from ephys_forever.gui.app import MainWindow

    window = MainWindow()
    window.show()
    window.open_path(iv_path)

    def next_sweep() -> None:
        spin = window.navigator.spin
        spin.setValue(spin.value() % spin.maximum() + 1)
        render(qapp, window.view)

    benchmark(next_sweep)
    window.close()


@pytest.mark.slow
@pytest.mark.gui
def test_draw_30_min_gap_free_under_target(benchmark: Any, qapp: Any) -> None:
    from ephys_forever.gui.views.trace_view import TraceView

    n = int(GAP_FREE_SECONDS * GAP_FREE_RATE)
    data = np.random.default_rng(0).normal(0.0, 5e-12, (1, n))
    rec = Recording(
        channels=(Channel("IN 0", "A", "current"),),
        sweeps=(Sweep(index=0, data=data, sample_rate=GAP_FREE_RATE),),
    )
    view = TraceView()
    view.resize(1600, 600)
    view.show()

    def draw() -> None:
        view.set_recording(rec)
        render(qapp, view)

    benchmark.pedantic(draw, rounds=3, iterations=1)
    assert benchmark.stats.stats.max < TARGET_S
    view.close()


@pytest.mark.slow
@pytest.mark.realdata
def test_read_30_min_gap_free_abf(benchmark: Any, gap_free_abf: Path) -> None:
    rec = benchmark.pedantic(read, args=(gap_free_abf,), rounds=3, iterations=1)
    assert rec.sweeps[0].n_samples == int(GAP_FREE_SECONDS * GAP_FREE_RATE)
    assert benchmark.stats.stats.median < TARGET_S


@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.gui
def test_open_and_draw_30_min_gap_free_abf(benchmark: Any, qapp: Any, gap_free_abf: Path) -> None:
    from ephys_forever.gui.app import MainWindow

    window = MainWindow()
    window.resize(1600, 900)
    window.show()

    def open_and_draw() -> None:
        window.open_path(gap_free_abf)
        render(qapp, window)

    benchmark.pedantic(open_and_draw, rounds=3, iterations=1)
    assert benchmark.stats.stats.median < TARGET_S
    window.close()
