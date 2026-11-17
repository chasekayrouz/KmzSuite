"""Clipped curve sections retain pyqtgraph pixels, data, and item ownership."""
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QRegion, QTransform
from PySide6.QtWidgets import QApplication, QWidget

from kmzviewer.gui.curves import ChromatogramCurve, ChromatogramItem
from kmzviewer.gui import plots


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def draw(curve, ratio=1, region=None, transform=None):
    image = QImage(640 * ratio, 480 * ratio, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(QColor('#e3f0ee'))
    painter = QPainter(image)
    try:
        painter.setClipRegion(region or QRegion(9, 13, 622, 453))
        painter.setTransform(transform or QTransform(1, 0, 0, -1, .123, 480.321))
        curve.paint(painter, None, None)
    finally:
        painter.end()
    return bytes(image.constBits())


@pytest.mark.parametrize('ratio', (1, 2))
@pytest.mark.parametrize('width', (1.7, 3.4))
def test_dense_stroke_and_fill_match(qapp, ratio, width):
    rng = np.random.default_rng(1207)
    for i in range(6):
        x = np.linspace(20., 620., 160 + i * 37)
        y = rng.uniform(30, 450, len(x))
        pen = pg.mkPen(QColor(0, 114, 178, 255 if i % 2 else 143), width=width)
        style = dict(pen=pen, brush=pg.mkBrush(0, 114, 178, 17), fillLevel=0,
                     connect='finite', antialias=True)
        fast, plain = ChromatogramCurve(x, y, **style), pg.PlotCurveItem(x, y, **style)
        assert draw(fast, ratio) == draw(plain, ratio)
        assert fast._stroke_bands is not None
        bands = fast._stroke_bands
        # Cached geometry is reused; data and transform changes discard it.
        assert draw(fast, ratio) == draw(plain, ratio)
        assert fast._stroke_bands is bands
        fast.setData(x, y[::-1])
        plain.setData(x, y[::-1])
        assert draw(fast, ratio) == draw(plain, ratio)
        assert fast._stroke_bands is not bands
        for curve in (fast, plain):
            curve.setPen(pg.mkPen('#cc3377', width=width))
        assert draw(fast, ratio) == draw(plain, ratio)
        shifted = QTransform(1, 0, 0, -1, 1.234, 479.123)
        assert draw(fast, ratio, transform=shifted) == draw(plain, ratio, transform=shifted)


@pytest.mark.parametrize('case', ('repeat', 'nan', 'mask', 'outside', 'dashed', 'noncosmetic', 'rotation', 'flat', 'aliased', 'large-origin'))
def test_unqualified_geometry_uses_original_paint(qapp, case):
    rng = np.random.default_rng(522)
    x = np.linspace(20., 620., 200)
    y = rng.uniform(30, 450, len(x))
    pen = pg.mkPen('#0072b2', width=1.7)
    connect = 'finite'
    transform = None
    if case == 'repeat': x[20] = x[19]
    if case == 'nan': y[20] = np.nan
    if case == 'mask': connect = rng.random(len(x)) > .2
    if case == 'outside': y *= 10
    if case == 'dashed': pen.setStyle(Qt.PenStyle.DashLine)
    if case == 'noncosmetic': pen.setCosmetic(False)
    if case == 'rotation': transform = QTransform().rotate(5)
    if case == 'flat': y[:] = 100
    if case == 'large-origin':
        x += 1e14
        transform = QTransform(1, 0, 0, -1, -1e14, 480)
    style = dict(pen=pen, brush=pg.mkBrush(0, 114, 178, 17), fillLevel=0, connect=connect, antialias=case != 'aliased')
    fast, plain = ChromatogramCurve(x, y, **style), pg.PlotCurveItem(x, y, **style)
    assert draw(fast, 2, transform=transform) == draw(plain, 2, transform=transform)
    assert fast._stroke_bands is None


def panel(qapp, fast):
    original = plots.ChromatogramItem
    plots.ChromatogramItem = ChromatogramItem if fast else pg.PlotDataItem
    try:
        panel = plots.ChromatogramPanel()
        panel.resize(743, 653)
        rng = np.random.default_rng(3801)
        x = np.linspace(0, 10, 220)
        rows = [(i, f'Trace {i}', ('#0072b2', '#d55e00', '#009e73')[i], x,
                 rng.uniform(0, 500, len(x)) - (50 if i == 1 else 0)) for i in range(3)]
        panel.set_traces(rows, units={0: 'counts', 1: 'mAU', 2: 'counts'})
        panel.show()
        qapp.processEvents()
        return panel
    finally:
        plots.ChromatogramItem = original


def rendered(panel, ratio, region):
    image = QImage(panel.size() * ratio, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(QColor('#123456'))
    QWidget.render(panel, image, QPoint(), region)
    return bytes(image.constBits())


def compare_panels(qapp):
    fast, plain = panel(qapp, True), panel(qapp, False)
    ratio = int(fast.devicePixelRatioF())
    fast_count = 0
    try:
        for mode, match, emphasis in (('overlay', False, set()), ('overlay', False, {1}),
                                      ('overlay', True, {0, 2}), ('stacked', True, set()),
                                      ('stacked', False, set())):
            for p in (fast, plain):
                p.set_mode(mode)
                p.set_match_y(match)
                p.set_emphasized(emphasis)
                p.set_x_range(.23, 8.71)
            qapp.processEvents()
            assert fast.grab().toImage() == plain.grab().toImage()
            fast_count += sum(c.curve._stroke_bands is not None for c in fast._curves.values())
            regions = [QRegion(0, 0, 743, 653), QRegion(53, 79, 133, 217),
                       QRegion(0, 0, 743, 170) | QRegion(13, 280, 590, 220)]
            for region in regions:
                assert rendered(fast, ratio, region) == rendered(plain, ratio, region)
            for tid in fast._curves:
                for a, b in zip(fast._curves[tid].curve.getData(), plain._curves[tid].curve.getData()):
                    assert np.array_equal(a, b)
        assert fast_count > 0
    finally:
        for p in (fast, plain):
            p.close()
            p.deleteLater()
        qapp.processEvents()


def test_panel_modes_axes_emphasis_and_partial_exposes(qapp):
    compare_panels(qapp)



def test_unqualified_versions_keep_pyqtgraph_curves(qapp, unqualified):
    """Other Qt versions draw pyqtgraph's own curves (and axes, without settling): same pixels."""
    images = []
    for qualified in (True, False):
        if not qualified:
            unqualified('Qt')  # Before building: qualification is fixed for a process.
        p = panel(qapp, True)
        try:
            p.set_x_range(.23, 8.71)
            qapp.processEvents()
            kind = ChromatogramCurve if qualified else pg.PlotCurveItem
            assert all(type(curve.curve) is kind for curve in p._curves.values())
            images.append(p.grab().toImage())
        finally:
            p.close()
            p.deleteLater()
            qapp.processEvents()
    assert images[0] == images[1]

def test_panel_retina_in_scaled_process():
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = f'''import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from PySide6.QtWidgets import QApplication
import test_curve_bands as t
app = QApplication([])
t.compare_panels(app)
print('ok')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env=dict(os.environ, QT_SCALE_FACTOR='2', QT_QPA_PLATFORM='offscreen'), timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith('ok')
