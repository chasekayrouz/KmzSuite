"""Kmz2Viewer spectrum interactions paint once; pre-paint axis pictures match paint."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication

from kmz2viewer.formula import MASS_H, MASS_O
from kmz2viewer.gui.app import MainWindow
from kmzcore import plot_paint
from kmzcore.plot_paint import finish_pending
from Tests.Viewer.mzml_factory import write_mzml

WATER = 2 * MASS_H + MASS_O


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class Paints(QObject):
    def __init__(self, widget):
        super().__init__()
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, _obj, event):  # noqa: N802
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def settled_paints(qapp, paints, action):
    for _ in range(3):
        qapp.processEvents()
    before = paints.count
    action()
    for _ in range(5):
        qapp.processEvents()
    return paints.count - before


def _scans(path):
    rng = np.random.default_rng(5)
    spectra = []
    # Intensity magnitudes change the Y tick text width, and so the plot width.
    for index, scale in enumerate((40.0, 7.0e6, 3.0e3, 9.0e8)):
        mz = np.sort(np.r_[rng.uniform(60.0, 295.0, 40), 300.0 - WATER, 300.0])
        spectra.append({"id": f"scan-{index}", "ms_level": 2, "centroid": True, "rt": 1.0 + index,
                        "tic": 1.0, "precursor_mz": 300.0, "charge": 1, "collision_energy": 20.0,
                        "mz": mz.tolist(), "intensity": (rng.uniform(0.05, 1.0, mz.size) * scale).tolist()})
    write_mzml(path, spectra)


def test_spectrum_interactions_paint_once(qapp, tmp_path):
    _scans(tmp_path / "A.mzML")
    window = MainWindow()
    window.resize(1200, 860)
    window.show()
    try:
        assert window.load_paths_sync([str(tmp_path / "A.mzML")]) == []
        paints = Paints(window.spectrum.plot.viewport())
        view = window.spectrum.plot.getViewBox()

        def select(row):
            window.table.selectRow(row)
            window.open_selected()

        actions = [lambda row=row: select(row) for row in range(4)]
        actions += [lambda: window._step_spectrum(-1), lambda: view.scaleBy(x=0.5), lambda: view.scaleBy(x=0.3),
                    lambda: view.translateBy(x=-20.0), lambda: view.translateBy(x=35.0),
                    window.spectrum.losses.toggle, window.spectrum.losses.toggle, window.spectrum.reset_range,
                    lambda: window.resize(1100, 820)]
        for action in actions:
            assert settled_paints(qapp, paints, action) == 1
        assert window.spectrum.label_texts() and window.spectrum.precursor_line.isVisible()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_pre_paint_axis_pictures_match_paint(qapp):
    x = np.linspace(100.0, 900.0, 300)
    plots = []
    for _ in range(2):
        plot = pg.PlotWidget()
        plot.resize(700, 400)
        plot.plot(x, np.abs(np.sin(x / 30.0)) * 4.0e7)
        plot.setXRange(150.0, 850.0, padding=0)
        plot.show()
        plots.append(plot)
    try:
        for _ in range(3):
            qapp.processEvents()
        for plot in plots:
            plot.setYRange(0.0, 1.0e9, padding=0)
        axes = (plots[0].getAxis("left"), plots[0].getAxis("bottom"))
        width = axes[0].width()
        finish_pending(plots[0], axes)
        assert all(axis.picture is not None for axis in axes)
        assert axes[0].width() != width  # new tick text laid out before any paint
        # The twin measures its axis while painting, then lays out and paints again.
        assert plots[1].getAxis("left").picture is None
        plots[1].grab()
        for _ in range(3):
            qapp.processEvents()
        images = [plot.grab().toImage() for plot in plots]
        assert images[0] == images[1]
        for name in ("left", "bottom"):
            pictures = [bytes(plot.getAxis(name).picture.data()) for plot in plots]
            assert pictures[0] == pictures[1]
        assert plots[0].getAxis("left").width() == plots[1].getAxis("left").width()
    finally:
        for plot in plots:
            plot.close()
            plot.deleteLater()
        qapp.processEvents()


def test_unqualified_versions_finish_nothing_before_the_paint(qapp, monkeypatch, unqualified):
    sent = []

    class Events:
        @staticmethod
        def sendPostedEvents(receiver, kind):  # noqa: N802
            sent.append(kind)
    monkeypatch.setattr(plot_paint, "QCoreApplication", Events)
    plot = pg.PlotWidget()
    plot.resize(700, 400)
    plot.plot(np.linspace(100.0, 900.0, 300), np.linspace(0.0, 4.0e7, 300))
    plot.show()
    try:
        for _ in range(3):
            qapp.processEvents()
        plot.setYRange(0.0, 1.0e9, padding=0)
        axes = (plot.getAxis("left"), plot.getAxis("bottom"))
        unqualified("pyqtgraph")
        finish_pending()  # CoMet without EIC tabs: Qt event delivery alone
        assert sent
        sent.clear()
        finish_pending(plot, axes)
        assert not sent and axes[0].picture is None  # measured by the paint
        unqualified("Qt")
        finish_pending()
        assert not sent
    finally:
        plot.close()
        plot.deleteLater()
        qapp.processEvents()


def test_unqualified_versions_fit_overlays_while_painting(qapp, tmp_path, monkeypatch, unqualified):
    _scans(tmp_path / "A.mzML")
    window = MainWindow()
    window.resize(1200, 860)
    window.show()
    try:
        assert window.load_paths_sync([str(tmp_path / "A.mzML")]) == []
        window.table.selectRow(0)
        window.open_selected()
        qapp.processEvents()
        prepared, scene = [], type(window.spectrum.plot.scene())
        original = scene.prepareForPaint
        monkeypatch.setattr(scene, "prepareForPaint", lambda self: prepared.append(self) or original(self))
        window.spectrum._redraw_overlays()
        assert prepared
        prepared.clear()
        unqualified("pyqtgraph")
        window.spectrum._redraw_overlays()
        assert not prepared and window.spectrum.label_texts()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
