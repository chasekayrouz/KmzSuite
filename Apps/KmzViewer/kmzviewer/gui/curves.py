"""Dense chromatogram strokes with disjoint device-pixel clips."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QImage, QPaintEngine, QPixmap, QTransform

from kmzcore.qualified import qualified

_SCALED = (QTransform.TransformationType.TxNone, QTransform.TransformationType.TxTranslate,
           QTransform.TransformationType.TxScale)


class ChromatogramCurve(pg.PlotCurveItem):
    def updateData(self, *args, **kwargs):  # noqa: N802
        self._stroke_bands = None
        self._band_key = None
        self._band_data_ok = None
        self._vertical_travel = None
        super().updateData(*args, **kwargs)

    def _bands(self, p):
        pen = self.opts['pen']
        if (self.xData is None or len(self.xData) < 128 or self.opts['stepMode']
                or self.opts['fillOutline'] or self.opts['shadowPen'] is not None or not self.opts['antialias']
                or self.opts['segmentedLineMode'] == 'on' or self._exportOpts is not False
                or not isinstance(self.opts['connect'], str) or self.opts['connect'] not in ('all', 'finite')
                or pen is None or not pen.isCosmetic() or pen.style() != Qt.PenStyle.SolidLine
                or pen.brush().style() != Qt.BrushStyle.SolidPattern
                or pen.joinStyle() != Qt.PenJoinStyle.BevelJoin or pen.capStyle() != Qt.PenCapStyle.SquareCap
                or not 1. <= pen.widthF() <= 4. or p.brush().style() != Qt.BrushStyle.NoBrush
                or p.paintEngine().type() != QPaintEngine.Type.Raster):
            return None
        if self._band_data_ok is None:
            self._band_data_ok = (np.isfinite(self.xData).all() and np.isfinite(self.yData).all()
                                  and np.all(self.xData[1:] > self.xData[:-1]))
            if self._band_data_ok:
                self._vertical_travel = float(np.abs(np.diff(self.yData)).sum())
        if not self._band_data_ok:
            return None
        transform = p.deviceTransform()
        if (transform.type() not in _SCALED or transform.m11() <= 0
                or max(abs(transform.dx()), abs(transform.dy())) > 1e7):
            return None
        if self._vertical_travel * abs(transform.m22()) < 4000.:
            return None
        world = p.worldTransform()
        p.setWorldTransform(QTransform())
        base = p.deviceTransform()
        p.setWorldTransform(world)
        ratio, dx, dy = base.m11(), base.dx(), base.dy()
        if (base.type() not in _SCALED or base.m22() != ratio
                or ratio not in (1., 2.) or not dx.is_integer() or not dy.is_integer()):
            return None
        # Qt clips outlines far beyond the raster device using path intersection,
        # which can round differently after splitting. Stay near the whole outline's
        # device bounds; partial exposes and extreme offscreen geometry fall back.
        region = p.paintEngine().systemClip()
        if not region.isEmpty():
            bounds = QRectF(region.boundingRect())
        elif isinstance(p.device(), (QImage, QPixmap)):
            bounds = QRectF(0, 0, p.device().width(), p.device().height())  # Already device pixels.
        else:
            return None  # Widget sizes are logical; never guess device bounds.
        path = self.getPath()
        control = transform.mapRect(path.controlPointRect())
        margin = 2. * pen.widthF() * ratio + 2.
        if (not bounds.adjusted(-32, -32, 32, 32).contains(control.adjusted(-margin, -margin, margin, margin))
                or max(abs(control.left()), abs(control.right()), abs(control.top()), abs(control.bottom())) > 16000):
            return None
        width = int(32 * ratio)
        first = math.floor(max(bounds.left(), control.left() - margin) / width) * width
        last = math.ceil(min(bounds.right(), control.right() + margin) / width) * width
        if last - first <= 2 * width:
            return None
        key = (transform, base, bounds, first, last, pen.widthF())
        if self._band_key != key:
            device_x = self.xData * transform.m11() + transform.dx()
            bands = []
            for left in range(first, last, width):
                # Two original neighbours retain the segment crossing the clip and
                # its join. New end caps lie beyond the pen/AA guard on either side.
                lo = max(0, int(np.searchsorted(device_x, left - margin)) - 2)
                hi = min(len(device_x), int(np.searchsorted(device_x, left + width + margin)) + 2)
                if hi - lo < 2:
                    continue
                section = pg.functions.arrayToQPath(self.xData[lo:hi], self.yData[lo:hi], connect='all', finiteCheck=False)
                clip = QRectF((left - dx) / ratio, (bounds.top() - dy) / ratio,
                              width / ratio, bounds.height() / ratio)
                bands.append((clip, section))
            self._stroke_bands = bands
            self._band_key = key
        return self._stroke_bands

    def paint(self, p, opt, widget):
        bands = self._bands(p)
        if bands is None:
            return super().paint(p, opt, widget)
        # Keep pyqtgraph's fill paths, chunking, order and painter state unchanged.
        pen = self.opts['pen']
        self.opts['pen'] = None
        try:
            super().paint(p, opt, widget)
        finally:
            self.opts['pen'] = pen
        p.setPen(pen)
        world = p.worldTransform()
        for clip, path in bands:
            p.save()
            p.setWorldTransform(QTransform())
            p.setClipRect(clip, Qt.ClipOperation.IntersectClip)
            p.setWorldTransform(world)
            p.drawPath(path)
            p.restore()


class ChromatogramItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        if not qualified("Qt", "PySide6", "pyqtgraph"):
            # Bands reproduce the qualified Qt raster engine and pyqtgraph curve; elsewhere keep pyqtgraph's own.
            super().__init__(*args, **kwargs)
            return
        super().__init__()
        old = self.curve
        old.sigClicked.disconnect(self.sigClicked)
        old.setParentItem(None)
        old.deleteLater()
        self.curve = ChromatogramCurve()
        self.curve.setParentItem(self)
        self.curve.stackBefore(self.scatter)  # Keep pyqtgraph's curve-below-symbols order.
        self.curve.sigClicked.connect(self.sigClicked)
        self.setCurveClickable(kwargs.get('clickable', False))
        self.setData(*args, **kwargs)
