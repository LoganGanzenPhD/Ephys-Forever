"""Stacked, x-linked trace plots: one row per visible channel."""

from __future__ import annotations

from collections.abc import Sequence

import pyqtgraph as pg

from ephys_forever.core.model import Recording
from ephys_forever.gui.views.envelope import Envelope

SELECTED_PEN = pg.mkPen("k", width=1)
LEFT_AXIS_WIDTH = 80
"""Fixed width for every y axis, so the rows line up and linked x ranges match exactly."""
POINTS_PER_PIXEL = 4
MIN_POINTS = 2048


def sweep_pen(index: int, count: int) -> pg.QtGui.QPen:
    """Distinct pen per sweep for overlays."""
    return pg.mkPen(pg.intColor(index, hues=max(count, 1), values=1, maxValue=200), width=1)


class TraceView(pg.GraphicsLayoutWidget):  # type: ignore[misc]
    """Plots of one recording.

    Data are plotted in SI units; the axes add SI prefixes (pA, mV, ms) for display.
    Curves are drawn from a min/max :class:`Envelope` at one to two bins per pixel of
    the visible range and redrawn on every zoom or pan, so a 30-minute gap-free channel
    costs the same to draw as a short sweep while keeping every peak. Mouse wheel zooms,
    dragging pans, and all rows share the time axis.
    """

    def __init__(self, parent: pg.QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.recording: Recording | None = None
        self.sweep = 0
        self.overlay = False
        self.channels: list[int] = []
        self.plots: dict[int, pg.PlotItem] = {}
        self.curves: dict[int, list[tuple[int, pg.PlotDataItem]]] = {}
        self._envelopes: dict[tuple[int, int], Envelope] = {}
        self._syncing = False

    # -- public API ---------------------------------------------------------------
    def set_recording(self, recording: Recording) -> None:
        """Show ``recording``: first sweep, all channels, no overlay, full range."""
        self.recording = recording
        self.sweep = 0
        self.overlay = False
        self.channels = list(range(recording.n_channels))
        self._envelopes.clear()
        self._rebuild()

    def show_sweep(self, index: int) -> None:
        """Select sweep ``index``; in overlay mode it is drawn on top in black."""
        rec = self._require()
        if not 0 <= index < rec.n_sweeps:
            raise IndexError(f"sweep {index} out of range 0..{rec.n_sweeps - 1}")
        self.sweep = index
        self._assign_curves()

    def set_overlay(self, on: bool) -> None:
        """Draw every sweep at once (True) or only the selected sweep (False)."""
        self.overlay = on
        self._assign_curves()

    def set_channels(self, channels: Sequence[int]) -> None:
        """Show only the given channel indices, in recording order."""
        rec = self._require()
        bad = [c for c in channels if not 0 <= c < rec.n_channels]
        if bad:
            raise IndexError(f"channels {bad} out of range 0..{rec.n_channels - 1}")
        self.channels = sorted(set(channels))
        self._rebuild()

    def auto_range(self) -> None:
        """Fit the time axis to a sweep and each y axis to its channel across all sweeps.

        Using every sweep keeps the scale fixed while stepping through sweeps, so sweeps
        can be compared by eye, as in an overlay.
        """
        rec = self._require()
        duration = rec.sweeps[0].duration
        for c, plot in self.plots.items():
            extents = [self._envelope(s, c).extent() for s in range(rec.n_sweeps)]
            lo, hi = min(e[0] for e in extents), max(e[1] for e in extents)
            if lo == hi:
                lo, hi = lo - 1e-12, hi + 1e-12
            plot.getViewBox().setRange(xRange=(0.0, duration), yRange=(lo, hi), padding=0.02)
        self._redraw()

    # -- internals ----------------------------------------------------------------
    def _require(self) -> Recording:
        if self.recording is None:
            raise RuntimeError("no recording loaded")
        return self.recording

    def _envelope(self, sweep: int, channel: int) -> Envelope:
        key = (sweep, channel)
        env = self._envelopes.get(key)
        if env is None:
            env = Envelope(self._require().sweeps[sweep].data[channel])
            self._envelopes[key] = env
        return env

    def _rebuild(self) -> None:
        self.clear()
        self.plots.clear()
        self.curves.clear()
        rec = self._require()
        for row, c in enumerate(self.channels):
            channel = rec.channels[c]
            plot = self.addPlot(row=row, col=0)
            plot.setLabel("left", channel.name, units=channel.unit)
            plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.disableAutoRange()
            plot.getViewBox().sigXRangeChanged.connect(self._sync_x)
            if row == 0:  # point budget follows the plot width
                plot.getViewBox().sigResized.connect(self._redraw)
            if row < len(self.channels) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "Time", units="s")
            self.plots[c] = plot
            self.curves[c] = []
        self._assign_curves(redraw=False)
        if self.plots:
            self.auto_range()

    def _sync_x(self, box: pg.ViewBox, x_range: tuple[float, float]) -> None:
        """Copy one row's time range to every other row, then redraw.

        Ranges are copied as numbers rather than with pyqtgraph's view linking, which
        maps through pixel geometry and drifts while the layout is still settling.
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            for plot in self.plots.values():
                other = plot.getViewBox()
                if other is not box:
                    other.setXRange(*x_range, padding=0)
        finally:
            self._syncing = False
        self._redraw()

    def _assign_curves(self, redraw: bool = True) -> None:
        """Match one curve item per drawn sweep, selected sweep last so it is on top."""
        rec = self._require()
        order = list(range(rec.n_sweeps)) if self.overlay else [self.sweep]
        if self.overlay:
            order.remove(self.sweep)
            order.append(self.sweep)
        for c, plot in self.plots.items():
            curves = self.curves[c]
            while len(curves) < len(order):
                curve = pg.PlotDataItem(skipFiniteCheck=True)
                plot.addItem(curve)
                curves.append((-1, curve))
            while len(curves) > len(order):
                plot.removeItem(curves.pop()[1])
            for k, s in enumerate(order):
                curve = curves[k][1]
                pen = (
                    SELECTED_PEN
                    if s == self.sweep or not self.overlay
                    else sweep_pen(s, len(order))
                )
                curve.setPen(pen)
                curves[k] = (s, curve)
        if redraw:
            self._redraw()

    def _redraw(self) -> None:
        """Refill every curve for the current x range at screen resolution."""
        if self.recording is None or not self.plots:
            return
        rate = self.recording.sample_rate
        first = next(iter(self.plots.values())).getViewBox()
        (x0, x1), _ = first.viewRange()
        budget = max(MIN_POINTS, int(first.width() * POINTS_PER_PIXEL))
        for c, curves in self.curves.items():
            for s, curve in curves:
                x, y = self._envelope(s, c).view(x0 * rate, x1 * rate, budget)
                curve.setData(x / rate, y)

    def displayed(self, channel: int, position: int = -1) -> tuple[int, pg.PlotDataItem]:
        """Sweep index and curve item at ``position`` in a channel's draw order."""
        return self.curves[channel][position]
