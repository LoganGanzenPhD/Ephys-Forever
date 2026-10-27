"""Stacked, x-linked trace plots: one row per visible channel."""

from __future__ import annotations

from collections.abc import Sequence

import pyqtgraph as pg

from ephys_forever.core.analysis.regions import Region
from ephys_forever.core.model import FloatArray, Recording
from ephys_forever.gui.views.envelope import Envelope

SELECTED_PEN = pg.mkPen("k", width=1)
FIT_PEN = pg.mkPen((220, 30, 30), width=2)
LEFT_AXIS_WIDTH = 80
"""Fixed width for every y axis, so the rows line up and linked x ranges match exactly."""
POINTS_PER_PIXEL = 4
MIN_POINTS = 2048
REGION_NAMES = ("baseline", "measure")
REGION_BRUSHES = {"baseline": (128, 128, 128, 40), "measure": (40, 110, 220, 40)}
DEFAULT_REGIONS = {"baseline": (0.0, 0.1), "measure": (0.4, 0.6)}
"""Cursor placement on a new recording, as fractions of the sweep duration."""


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

    Two cursor regions, ``"baseline"`` (grey) and ``"measure"`` (blue), span every row.
    Dragging one in any row moves it in all rows and emits ``regionsChanged``.
    """

    regionsChanged = pg.QtCore.Signal(str)

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
        self.regions: dict[str, Region] = {}
        self.region_items: dict[int, dict[str, pg.LinearRegionItem]] = {}
        self._baseline_visible = True
        self.fits: dict[int, dict[int, tuple[FloatArray, FloatArray]]] = {}
        self.fit_items: list[tuple[int, pg.PlotDataItem]] = []

    # -- public API ---------------------------------------------------------------
    def set_recording(self, recording: Recording) -> None:
        """Show ``recording``: first sweep, all channels, no overlay, full range."""
        self.recording = recording
        self.sweep = 0
        self.overlay = False
        self.channels = list(range(recording.n_channels))
        self._envelopes.clear()
        self.fits.clear()
        duration = recording.sweeps[0].duration
        for name in REGION_NAMES:
            kept = self.regions.get(name)
            if kept is None or kept.stop > duration - 1.0 / recording.sample_rate:
                lo, hi = DEFAULT_REGIONS[name]
                self.regions[name] = Region(lo * duration, hi * duration)
        self._rebuild()

    def replace_data(self, recording: Recording) -> None:
        """Show new data of the same shape (for example after filtering), keeping the
        selected sweep, overlay, visible channels, cursors and zoom."""
        old = self._require()
        if (recording.n_sweeps, recording.n_channels) != (old.n_sweeps, old.n_channels):
            raise ValueError("replace_data needs the same sweeps and channels")
        self.recording = recording
        self._envelopes.clear()
        self._redraw()

    def set_fits(self, channel: int, fits: dict[int, tuple[FloatArray, FloatArray]]) -> None:
        """Draw fitted curves ``{sweep: (t, y)}`` over a channel's traces, in red."""
        self.fits[channel] = dict(fits)
        self._show_fits()

    def clear_fits(self) -> None:
        self.fits.clear()
        self._show_fits()

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

    def set_region(self, name: str, region: Region) -> None:
        """Move cursor region ``name`` ("baseline" or "measure") in every row."""
        if name not in REGION_NAMES:
            raise KeyError(f"no cursor region {name!r}; expected one of {REGION_NAMES}")
        self.regions[name] = region
        self._syncing = True
        try:
            for items in self.region_items.values():
                items[name].setRegion((region.start, region.stop))
        finally:
            self._syncing = False
        self.regionsChanged.emit(name)

    def set_baseline_visible(self, on: bool) -> None:
        """Show or hide the baseline region."""
        self._baseline_visible = on
        for items in self.region_items.values():
            items["baseline"].setVisible(on)

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
        self.region_items.clear()
        self.fit_items.clear()
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
            self.region_items[c] = {name: self._add_region(plot, name) for name in REGION_NAMES}
        self._assign_curves(redraw=False)
        if self.plots:
            self.auto_range()

    def _add_region(self, plot: pg.PlotItem, name: str) -> pg.LinearRegionItem:
        region = self.regions[name]
        item = pg.LinearRegionItem(
            values=(region.start, region.stop), brush=REGION_BRUSHES[name], swapMode="sort"
        )
        item.setZValue(-10)
        item.setToolTip(f"{name.capitalize()} region: drag the edges or the middle")
        item.setVisible(name != "baseline" or self._baseline_visible)
        item.sigRegionChanged.connect(lambda it, n=name: self._on_region_item(n, it))
        plot.addItem(item, ignoreBounds=True)
        return item

    def _on_region_item(self, name: str, item: pg.LinearRegionItem) -> None:
        """Copy a dragged region to every row and record it."""
        if self._syncing:
            return
        start, stop = (float(v) for v in item.getRegion())
        if stop <= start:  # collapsed by dragging; keep the last valid region
            return
        self.regions[name] = Region(start, stop)
        self._syncing = True
        try:
            for items in self.region_items.values():
                if items[name] is not item:
                    items[name].setRegion((start, stop))
        finally:
            self._syncing = False
        self.regionsChanged.emit(name)

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
        self._show_fits()
        if redraw:
            self._redraw()

    def _show_fits(self) -> None:
        """One red curve per drawn sweep that has a fit, on top of the traces."""
        for c, item in self.fit_items:
            if c in self.plots:
                self.plots[c].removeItem(item)
        self.fit_items.clear()
        for c, fits in self.fits.items():
            if c not in self.plots:
                continue
            for s, _ in self.curves[c]:
                if s in fits:
                    t, y = fits[s]
                    item = pg.PlotDataItem(t, y, pen=FIT_PEN)
                    item.setZValue(10)
                    self.plots[c].addItem(item)
                    self.fit_items.append((c, item))

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
