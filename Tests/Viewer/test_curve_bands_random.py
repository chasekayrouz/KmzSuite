"""Proposed addition: seeded corpus that fails when the band guard is weakened."""
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtGui import QColor, QImage, QPainter, QTransform
from PySide6.QtWidgets import QApplication

from kmzviewer.gui.curves import ChromatogramCurve


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def traces(rng):
    # Irregular spacing: sparse zero baselines, long segments across bands, clusters.
    for i in range(16):
        n = int(rng.integers(130, 600))
        x = np.unique(np.concatenate([np.sort(rng.uniform(20, 880, n)), np.arange(1, 28) * 32 + rng.uniform(-8, 8)]))
        y = np.where(rng.random(len(x)) < .5, 0., rng.uniform(0, 380, len(x)))
        if i % 4 == 0:
            keep = np.concatenate([np.arange(len(x) // 3), np.arange(len(x) // 3, len(x), 9)])
            x, y = x[keep], y[keep]
        yield x, y, (1.7, 3.4)[i % 2], (1, 2)[(i // 2) % 2], (255, 143)[(i // 4) % 2]


def draw(cls, x, y, width, ratio, alpha):
    item = cls(x, y, pen=pg.mkPen(QColor(0, 114, 178, alpha), width=width), brush=pg.mkBrush(0, 114, 178, 17),
               fillLevel=0., connect='finite', antialias=True)
    image = QImage(900 * ratio, 420 * ratio, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(ratio)
    image.fill(QColor('#e3f0ee'))
    painter = QPainter(image)
    painter.setTransform(QTransform(1, 0, 0, -1, .37, 400.37))
    item.paint(painter, None, None)
    painter.end()
    return bytes(image.constBits()), getattr(item, '_stroke_bands', None) is not None


def test_irregular_traces_match_original_painter(qapp):
    used = 0
    for i, (x, y, width, ratio, alpha) in enumerate(traces(np.random.default_rng(20261115))):
        fast, banded = draw(ChromatogramCurve, x, y, width, ratio, alpha)
        used += banded
        assert fast == draw(pg.PlotCurveItem, x, y, width, ratio, alpha)[0], f'trace {i}'
    assert used >= 12
