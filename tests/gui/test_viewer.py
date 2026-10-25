"""Viewer smoke tests: open, browse sweeps and channels, overlay, linked zoom and pan."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ephys_forever.core.io.registry import read
from ephys_forever.gui.app import MainWindow

pytestmark = pytest.mark.gui

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture
def window(qapp: Any) -> Iterator[MainWindow]:
    win = MainWindow()
    win.show()
    qapp.processEvents()
    yield win
    win.close()


@pytest.fixture
def iv_path(real_data_file: Callable[[str], Path]) -> Path:
    return real_data_file("vclamp_iv_steps_001.abf")


def assert_shows(window: MainWindow, channel: int, sweep: int, position: int = -1) -> None:
    """The curve draws ``sweep``: only its samples, including its extremes."""
    rec = window.recording
    assert rec is not None
    shown, curve = window.view.displayed(channel, position)
    assert shown == sweep
    _, y = curve.getOriginalDataset()
    data = rec.sweeps[sweep].data[channel]
    assert y is not None and np.isin(y, data).all()
    assert (y.min(), y.max()) == (data.min(), data.max())


@pytest.mark.realdata
def test_open_shows_every_channel_of_first_sweep(window: MainWindow, iv_path: Path) -> None:
    assert window.open_path(iv_path)
    rec = window.recording
    assert rec is not None
    assert sorted(window.view.plots) == [0, 1, 2]
    for c in range(3):
        assert len(window.view.curves[c]) == 1
        assert_shows(window, c, 0)
    assert window.navigator.count_label.text() == "of 9"
    assert window.view.plots[0].getAxis("left").labelUnits == "A"
    assert window.view.plots[1].getAxis("left").labelUnits == "V"


@pytest.mark.realdata
def test_fit_shows_whole_sweep_and_every_sweeps_range(
    qapp: Any, window: MainWindow, iv_path: Path
) -> None:
    window.open_path(iv_path)
    qapp.processEvents()
    rec = window.recording
    assert rec is not None
    for c, plot in window.view.plots.items():
        (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
        assert x0 <= 0.0 and x1 >= rec.sweeps[0].duration
        assert y0 <= min(s.data[c].min() for s in rec.sweeps)
        assert y1 >= max(s.data[c].max() for s in rec.sweeps)


@pytest.mark.realdata
def test_sweep_navigation(window: MainWindow, iv_path: Path) -> None:
    window.open_path(iv_path)
    rec = window.recording
    assert rec is not None
    window.navigator.next_button.click()
    window.navigator.next_button.click()
    assert window.view.sweep == 2
    assert_shows(window, 0, 2)
    window.navigator.prev_button.click()
    assert window.view.sweep == 1
    window.navigator.spin.setValue(9)
    assert_shows(window, 1, 8)


@pytest.mark.realdata
def test_overlay_draws_every_sweep_with_selected_last(window: MainWindow, iv_path: Path) -> None:
    window.open_path(iv_path)
    rec = window.recording
    assert rec is not None
    window.navigator.spin.setValue(4)
    window.navigator.overlay_box.setChecked(True)
    assert all(len(curves) == 9 for curves in window.view.curves.values())
    assert_shows(window, 0, 3, -1)
    assert sorted(s for s, _ in window.view.curves[0]) == list(range(9))
    window.navigator.overlay_box.setChecked(False)
    assert all(len(curves) == 1 for curves in window.view.curves.values())


@pytest.mark.realdata
def test_hiding_channels(window: MainWindow, iv_path: Path) -> None:
    window.open_path(iv_path)
    window.channel_list.set_checked(1, False)
    assert sorted(window.view.plots) == [0, 2]
    window.channel_list.set_checked(1, True)
    assert sorted(window.view.plots) == [0, 1, 2]


@pytest.mark.realdata
def test_zoom_and_pan_are_linked_across_channels(
    qapp: Any, window: MainWindow, iv_path: Path
) -> None:
    window.open_path(iv_path)
    plots = window.view.plots
    plots[0].getViewBox().setXRange(0.10, 0.12, padding=0)
    qapp.processEvents()
    for c in (1, 2):
        lo, hi = plots[c].getViewBox().viewRange()[0]
        assert (lo, hi) == pytest.approx((0.10, 0.12))
    plots[2].getViewBox().translateBy(x=0.01)
    qapp.processEvents()
    assert plots[0].getViewBox().viewRange()[0] == pytest.approx([0.11, 0.13])


@pytest.mark.realdata
def test_zooming_in_draws_raw_samples(qapp: Any, window: MainWindow, iv_path: Path) -> None:
    window.open_path(iv_path)
    rec = window.recording
    assert rec is not None
    window.view.plots[0].getViewBox().setXRange(0.100, 0.105, padding=0)
    qapp.processEvents()
    x, y = window.view.displayed(0)[1].getOriginalDataset()
    assert x is not None and y is not None
    i0 = round(float(x[0]) * rec.sample_rate)
    np.testing.assert_array_equal(y, rec.sweeps[0].data[0, i0 : i0 + y.size])
    assert x[0] <= 0.100 and x[-1] >= 0.105


def test_gap_free_file_disables_sweep_controls(window: MainWindow) -> None:
    assert window.open_path(SYNTHETIC / "epsc_train_known_times.atf")
    assert not window.navigator.next_button.isEnabled()
    assert not window.navigator.overlay_box.isEnabled()
    assert len(window.view.curves[0]) == 1


def test_open_failure_reports_and_keeps_current_file(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "show_error", lambda t, m: errors.append((t, m)))
    good = SYNTHETIC / "exp_decay_single.atf"
    window.open_path(good)
    bad = tmp_path / "junk.dat"
    bad.write_bytes(b"not a recording")
    assert not window.open_path(bad)
    assert errors and "junk.dat" in errors[0][0]
    assert window.recording is not None
    assert window.recording.metadata["source_path"] == str(good)


def test_set_recording_directly(window: MainWindow) -> None:
    rec = read(SYNTHETIC / "exp_decay_double.atf")
    window.set_recording(rec, "x")
    assert window.windowTitle().startswith("x")
    assert "4 sweeps" in window.statusBar().currentMessage()
