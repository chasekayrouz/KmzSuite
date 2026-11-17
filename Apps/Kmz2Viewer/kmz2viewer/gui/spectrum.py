"""Interactive MS2 spectrum, peak labels, precursor mark, and Δformula connectors."""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kmz2viewer.annotate import place_annotations
from kmz2viewer.differences import SpectrumAnalysis
from kmz2viewer.model import Ms2Scan
from kmzcore.plot_labels import add_peak_labels
from kmzcore.qualified import qualified
from kmzcore.ranges import padded_y, values_in_window
from kmzcore.theme import SPECTRUM, configure_plots

configure_plots()


class SpectrumView(QWidget):
    previous_requested = Signal()
    next_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scan: Ms2Scan | None = None
        self._analysis: SpectrumAnalysis | None = None
        self._show_losses = True
        self._user_x: tuple[float, float] | None = None
        self._guard = False
        self._label_items: list[pg.TextItem] = []
        self._loss_items: list = []
        self._label_font = QFont()
        self._label_font.setPointSize(9)

        self.caption = QLabel("Open a row to inspect an MS2 scan.")
        self.caption.setWordWrap(True)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.losses = QCheckBox("Show neutral losses")
        self.losses.setChecked(True)
        self.losses.toggled.connect(self._toggle_losses)
        self.settings_button = QPushButton("Formula settings")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.full_button = QPushButton("Full m/z range")
        self.previous_button.clicked.connect(self.previous_requested.emit)
        self.next_button.clicked.connect(self.next_requested.emit)
        self.full_button.clicked.connect(self.reset_range)

        header = QHBoxLayout()
        header.addWidget(self.previous_button)
        header.addWidget(self.next_button)
        header.addWidget(self.losses)
        header.addWidget(self.settings_button)
        header.addWidget(self.full_button)
        header.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.getViewBox().setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.setLabel("bottom", "m/z")
        self.plot.setLabel("left", "Intensity")
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.sigXRangeChanged.connect(self._on_x)
        self.plot.getViewBox().sigResized.connect(self._redraw_overlays)
        self.plot.scene().sigMouseMoved.connect(self._moved)
        self.precursor_line: pg.InfiniteLine | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.caption)
        layout.addWidget(self.detail)
        layout.addLayout(header)
        layout.addWidget(self.plot, stretch=1)

    def set_navigation(self, has_previous: bool, has_next: bool) -> None:
        self.previous_button.setEnabled(has_previous)
        self.next_button.setEnabled(has_next)

    def show_scan(self, scan: Ms2Scan, analysis: SpectrumAnalysis | None, *, keep_window: bool) -> None:
        self._scan = scan
        self._analysis = analysis
        self.caption.setText(_caption(scan))
        self.detail.setText(_assumption(scan))
        self._label_items.clear()
        self._loss_items.clear()
        self.plot.clear()
        self.precursor_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#b42318", width=1.2, style=Qt.PenStyle.DashLine)
        )
        self.precursor_line.setZValue(8)
        self.plot.addItem(self.precursor_line)
        pen = pg.mkPen(SPECTRUM, width=1.1)
        if scan.mz.size and scan.centroid:
            sticks_x, sticks_y = _sticks(scan.mz, scan.intensity)
            self.plot.plot(sticks_x, sticks_y, pen=pen, connect="pairs")
        elif scan.mz.size:
            self.plot.plot(scan.mz, scan.intensity, pen=pen)
        self.plot.setLabel("bottom", "m/z")
        self.plot.setLabel("left", "Intensity")
        self._place_precursor()
        self._guard = True
        try:
            if keep_window and self._user_x is not None:
                self.plot.setXRange(self._user_x[0], self._user_x[1], padding=0)
            else:
                self._user_x = None
                self._set_full_x()
            self._fit_y()
        finally:
            self._guard = False
        self._redraw_overlays()

    def reset_range(self) -> None:
        self._user_x = None
        if self._scan is None or self._scan.mz.size == 0:
            return
        self._guard = True
        try:
            self._set_full_x()
            self._fit_y()
        finally:
            self._guard = False
        self._redraw_overlays()

    def set_show_losses(self, enabled: bool) -> None:
        self._show_losses = enabled
        if self.losses.isChecked() != enabled:
            self.losses.setChecked(enabled)
        self._fit_y()
        self._redraw_overlays()

    def label_texts(self) -> list[str]:
        return [item.toPlainText() for item in self._label_items]

    def formula_texts(self) -> list[str]:
        return [item.toPlainText() for item in self._loss_items if isinstance(item, pg.TextItem)]

    def _toggle_losses(self, enabled: bool) -> None:
        self._show_losses = enabled
        if self._scan is not None:
            self._guard = True
            try:
                self._fit_y()
            finally:
                self._guard = False
            self._redraw_overlays()

    def _on_x(self, _plot, _range) -> None:
        if self._guard or self._scan is None:
            return
        x0, x1 = self.plot.viewRange()[0]
        self._user_x = (float(x0), float(x1))
        self._guard = True
        try:
            self._fit_y()
        finally:
            self._guard = False
        self._redraw_overlays()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)

    def _set_full_x(self) -> None:
        scan = self._scan
        if scan is None:
            return
        finite = scan.mz[np.isfinite(scan.mz)] if scan.mz.size else scan.mz
        if finite.size == 0:
            return
        xmin = float(finite.min())
        xmax = float(finite.max())
        if scan.precursor_mz is not None and np.isfinite(scan.precursor_mz):
            xmin = min(xmin, float(scan.precursor_mz))
            xmax = max(xmax, float(scan.precursor_mz))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        pad = (xmax - xmin) * 0.02
        self.plot.setXRange(xmin - pad, xmax + pad, padding=0)

    def _fit_y(self) -> None:
        scan = self._scan
        if scan is None or scan.mz.size == 0:
            return
        x0, x1 = self.plot.viewRange()[0]
        visible = values_in_window(scan.mz, scan.intensity, float(x0), float(x1))
        ymin, ymax = padded_y(visible)
        if self._show_losses and self._analysis is not None and any(pair.best for pair in self._analysis.pairs):
            span = ymax - ymin
            ymax = ymax + span * 0.45
        if ymin > 0:
            ymin = 0.0
        self.plot.setYRange(ymin, ymax, padding=0)

    def _place_precursor(self) -> None:
        scan = self._scan
        if scan is None or scan.precursor_mz is None or not np.isfinite(scan.precursor_mz):
            self.precursor_line.setVisible(False)
            return
        self.precursor_line.setPos(float(scan.precursor_mz))
        self.precursor_line.setVisible(True)

    def _redraw_overlays(self) -> None:
        if self._guard:
            return
        self._place_overlays()
        if not qualified("Qt", "PySide6", "pyqtgraph"):
            return  # Other versions keep pyqtgraph's paint-time updates.
        texts = [item for item in self._label_items + self._loss_items if isinstance(item, pg.TextItem)]
        if texts or self.precursor_line is not None:
            # A new TextItem otherwise fits its transform, and the precursor line
            # its bounds for a changed view, during the next paint, which
            # schedules a second full repaint of the same frame.
            self.plot.scene().prepareForPaint()
            for item in texts:
                item.updateTransform()
            if self.precursor_line is not None:
                self.precursor_line.boundingRect()

    def _place_overlays(self) -> None:
        for item in self._label_items + self._loss_items:
            self.plot.removeItem(item)
        self._label_items.clear()
        self._loss_items.clear()
        scan = self._scan
        if scan is None or scan.mz.size == 0:
            return
        self._guard = True
        try:
            self._fit_y()
            self._label_items = add_peak_labels(self.plot, scan.mz, scan.intensity, scan.centroid, self._label_font)
        finally:
            self._guard = False
        x0, x1 = [float(value) for value in self.plot.viewRange()[0]]
        y0, y1 = [float(value) for value in self.plot.viewRange()[1]]
        width = float(self.plot.getViewBox().width())
        if width < 30:
            width = float(self.plot.width())
        height = float(self.plot.getViewBox().height())
        if height < 30:
            height = float(self.plot.height())
        if not self._show_losses or self._analysis is None or width < 30 or height < 30 or y1 <= y0:
            return
        placed = place_annotations(self._analysis.pairs, x0, x1, width)
        pixel = (y1 - y0) / height
        for annotation in placed:
            pair = annotation.pair
            best = pair.best
            if best is None:
                continue
            y_line = y1 - (annotation.tier + 1) * 18.0 * pixel
            left_y = _endpoint_y(scan, pair.left_mz, pair.left_intensity)
            right_y = _endpoint_y(scan, pair.right_mz, pair.right_intensity)
            xs = [pair.left_mz, pair.left_mz, pair.right_mz, pair.right_mz]
            ys = [left_y, y_line, y_line, right_y]
            curve = self.plot.plot(xs, ys, pen=pg.mkPen("#6d4c9a", width=1.1))
            curve.setZValue(15)
            self._loss_items.append(curve)
            text = pg.TextItem(best.formula, color="#3d2a55", anchor=(0.5, 1.05))
            text.setFont(self._label_font)
            text.setPos((pair.left_mz + pair.right_mz) / 2.0, y_line)
            text.setZValue(21)
            text.setToolTip(_tooltip(pair))
            self.plot.addItem(text)
            self._loss_items.append(text)

    def _moved(self, scene_pos) -> None:
        scan = self._scan
        if scan is None or scan.mz.size == 0:
            return
        view = self.plot.getViewBox()
        if not view.sceneBoundingRect().contains(scene_pos):
            return
        point = view.mapSceneToView(scene_pos)
        index = int(np.argmin(np.abs(scan.mz - point.x())))
        message = f"m/z {scan.mz[index]:.4f}    {scan.intensity[index]:.6g}"
        tip = _annotation_under(self._analysis if self._show_losses else None, float(point.x()), float(point.y()), scan)
        if tip:
            message = tip
        self.detail.setText(tip or _assumption(scan))
        self.plot.setToolTip(message)


def _endpoint_y(scan: Ms2Scan, mz: float, intensity: float) -> float:
    if intensity <= 0:
        return 0.0
    return float(intensity)


def _sticks(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = mz.size
    x_values = np.empty(count * 2, dtype=np.float64)
    y_values = np.empty(count * 2, dtype=np.float64)
    x_values[0::2] = mz
    x_values[1::2] = mz
    y_values[0::2] = 0.0
    y_values[1::2] = intensity
    return x_values, y_values


def _caption(scan: Ms2Scan) -> str:
    charge = "" if scan.charge is None else f"    z {scan.charge}"
    energy = "" if scan.collision_energy is None else f"    CE {scan.collision_energy:g}"
    precursor = "precursor —" if scan.precursor_mz is None else f"precursor {scan.precursor_mz:.4f}"
    mode = "centroid" if scan.centroid else "profile"
    rt = "RT —" if not np.isfinite(scan.rt_min) else f"RT {scan.rt_min:.4f} min"
    return (
        f"{scan.file_name}    {scan.scan_id}    {rt}    {precursor}{charge}{energy}    MS2 {mode}"
    )


def _assumption(scan: Ms2Scan) -> str:
    if scan.charge is not None and scan.charge > 1:
        return f"Precursor charge is {scan.charge}. Precursor-to-fragment losses are not assigned."
    return "Δm/z is treated as a neutral mass, the singly charged assumption."


def _tooltip(pair) -> str:
    best = pair.best
    if best is None:
        return ""
    return (
        f"Formula: {best.formula}\n"
        f"Observed Δm/z: {pair.delta:.4f}\n"
        f"Theoretical mass: {best.theoretical:.5f}\n"
        f"Error: {best.ppm_error:+.1f} ppm\n"
        f"Peaks: {pair.left_mz:.4f} → {pair.right_mz:.4f}\n"
        f"Precursor relationship: {'yes' if pair.involves_precursor else 'no'}"
    )


def _annotation_under(analysis: SpectrumAnalysis | None, mz: float, intensity: float, scan: Ms2Scan) -> str:
    if analysis is None:
        return ""
    nearest = None
    nearest_distance = None
    for pair in analysis.pairs:
        if pair.best is None:
            continue
        if mz < pair.left_mz or mz > pair.right_mz:
            continue
        center = (pair.left_mz + pair.right_mz) / 2.0
        distance = abs(mz - center)
        if nearest_distance is None or distance < nearest_distance:
            nearest = pair
            nearest_distance = distance
    if nearest is None:
        return ""
    return _tooltip(nearest).replace("\n", "    ")
