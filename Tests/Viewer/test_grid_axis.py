"""GridAxis replays pyqtgraph's axis pictures under an explicit clip: same pixels."""
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPicture, QRegion
from PySide6.QtWidgets import QApplication, QWidget

import kmzviewer.gui.axes as axes
from kmzviewer.gui.axes import GridAxis


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def stack(qapp, grid_axes):
    view = pg.GraphicsLayoutWidget()
    view.resize(533, 611)  # fractional pane positions
    view.ci.setSpacing(0)
    x = np.linspace(0., 10., 300)
    for i in range(3):
        plot = pg.PlotItem(axisItems={side: GridAxis(side) for side in ("left", "bottom")} if grid_axes else None,
                           enableMenu=False)
        plot.showGrid(x=True, y=True, alpha=(.16, .10, .4)[i])
        # A pane without values has zero-length (point) ticks; others draw alpha grids.
        plot.getAxis("left").setStyle(showValues=bool(i), tickLength=-5 if i else 0)
        plot.setLabel("left", "Counts" if i == 1 else "")
        plot.plot(x, np.sin(x * (i + 2)) * 10. ** (3 * i), pen=pg.mkPen("#0072b2", width=1.7))
        plot.setXRange(.37, 9.13, padding=0)
        plot.setYRange(-1.3 * 10. ** (3 * i), 1.1 * 10. ** (3 * i), padding=0)
        view.ci.addItem(plot, row=i, col=0)
    view.show()
    for _ in range(3):
        qapp.processEvents()
        view.grab()
    return view


def rendered(view, region):
    image = QImage(view.size(), QImage.Format.Format_RGB32)
    image.fill(QColor("#123456"))
    QWidget.render(view, image, QPoint(), region)  # the widget paint path, as grab()/backing store
    return bytes(image.constBits())


REGIONS = {
    "full": QRegion(0, 0, 533, 611),
    "inner rect": QRegion(41, 17, 300, 260),
    "two rects": QRegion(0, 0, 533, 180) | QRegion(57, 180, 400, 60),
    "strips": sum((QRegion(30 + 70 * k, 0, 9, 611) for k in range(7)), QRegion()),
    "beyond view": QRegion(-20, 500, 600, 200),
}


def test_grid_lines_match_axisitem_pixels_for_paint_regions(qapp, monkeypatch):
    exposed = []
    original = axes._exposed_rects
    monkeypatch.setattr(axes, "_exposed_rects", lambda *a, **k: exposed.append(original(*a, **k)) or exposed[-1])
    fast, plain = stack(qapp, True), stack(qapp, False)
    try:
        assert fast.grab().toImage() == plain.grab().toImage()
        for name, region in REGIONS.items():
            assert rendered(fast, region) == rendered(plain, region), name
        # Single-rect, multi-rect and too-many-rect (plain replay) clips were all exercised.
        counts = {None if rects is None else len(rects) for rects in exposed}
        assert {None, 1, 2} <= counts
    finally:
        for view in (fast, plain):
            view.close()
            view.deleteLater()
        qapp.processEvents()


def test_existing_clip_and_picture_devices_replay_unchanged(qapp):
    fast, plain = stack(qapp, True), stack(qapp, False)
    try:
        images = []
        for view in (fast, plain):
            picture = QPicture()
            painter = QPainter(picture)
            view.scene().render(painter)  # non-raster engine
            painter.end()
            image = QImage(view.size(), QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(QColor("white"))
            painter = QPainter(image)
            painter.setClipRect(QRect(13, 9, 400, 500))  # pre-existing painter clip
            picture.play(painter)
            painter.end()
            images.append(image)
        assert images[0] == images[1]
    finally:
        for view in (fast, plain):
            view.close()
            view.deleteLater()
        qapp.processEvents()



def test_unqualified_versions_paint_as_axisitem(qapp, unqualified):
    plain = stack(qapp, grid_axes=False)
    unqualified("pyqtgraph")
    fallback = stack(qapp, grid_axes=True)
    try:
        grid = [fallback.ci.getItem(i, 0).getAxis(side) for i in range(3) for side in ("left", "bottom")]
        assert all(type(axis) is GridAxis and axis._lines is None and axis._label_state is None for axis in grid)
        for region in REGIONS.values():
            assert rendered(fallback, region) == rendered(plain, region)
    finally:
        for view in (plain, fallback):
            view.close()
            view.deleteLater()
        qapp.processEvents()

def test_label_html_follows_axisitem(qapp):
    view = pg.GraphicsLayoutWidget()
    fast = pg.PlotItem(axisItems={"left": GridAxis("left")})
    plain = pg.PlotItem()
    view.ci.addItem(fast, row=0, col=0)
    view.ci.addItem(plain, row=1, col=0)
    view.show()
    qapp.processEvents()
    try:
        steps = [lambda p: p.setLabel("left", "Counts"), lambda p: p.setYRange(0, 3e7, padding=0),
                 lambda p: p.setYRange(0, 3e7, padding=0), lambda p: p.setYRange(0, 10, padding=0),
                 lambda p: p.setLabel("left", "Counts", units="V"), lambda p: p.setYRange(0, 2e-4, padding=0),
                 lambda p: p.getAxis("left").enableAutoSIPrefix(False), lambda p: p.setLabel("left", "")]
        for step in steps:
            step(fast)
            step(plain)
            assert fast.getAxis("left").label.toHtml() == plain.getAxis("left").label.toHtml()
        # Text changed behind the axis's back is replaced by the next identical label.
        for plot in (fast, plain):
            plot.getAxis("left").label.setPlainText("stale")
            plot.setLabel("left", "")
        assert "stale" not in fast.getAxis("left").label.toPlainText()
        assert fast.getAxis("left").label.toHtml() == plain.getAxis("left").label.toHtml()
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_pixel_ratio_two_matches_in_a_scaled_process():
    """Retina-like painting (device pixel ratio 2) takes the same exact path."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QWidget
import kmzviewer.gui.axes as axes
import test_grid_axis as t
app = QApplication([])
seen = []
original = axes._exposed_rects
axes._exposed_rects = lambda *a, **k: seen.append(original(*a, **k)) or seen[-1]
fast, plain = t.stack(app, True), t.stack(app, False)
assert fast.devicePixelRatioF() == 2.0
def rendered(view, region):
    image = QImage(view.size() * 2, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(2.0)
    image.fill(QColor("#123456"))
    QWidget.render(view, image, QPoint(), region)
    return bytes(image.constBits())
assert fast.grab().toImage() == plain.grab().toImage()
for name, region in t.REGIONS.items():
    assert rendered(fast, region) == rendered(plain, region), name
assert {{1, 2}} <= {{len(r) for r in seen if r is not None}}
print("ok")
"""
    env = dict(os.environ, QT_SCALE_FACTOR="2", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.stdout.strip().endswith("ok"), result.stderr[-2000:]
