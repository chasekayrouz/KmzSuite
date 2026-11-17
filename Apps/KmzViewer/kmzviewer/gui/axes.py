"""Axis items that let Qt draw grid lines through its direct-pixel path."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPaintEngine, QPainter, QPen, QPicture, QTransform

from kmzcore.qualified import qualified

_RASTER = QPaintEngine.Type.Raster
_SCALED = (QTransform.TransformationType.TxNone, QTransform.TransformationType.TxTranslate,
           QTransform.TransformationType.TxScale)


def _exposed_rects(p: QPainter, bounds, limit=4):
    """Logical rects equal to the paint event's clip near ``bounds``, or None.

    With only a system (paint-event) clip, Qt's raster engine blends aliased
    cosmetic lines one span per pixel. An explicit painter clip of the same
    device pixels selects QCosmeticStroker's direct pixel writes: the same
    pixels and values at ~10x less cost. Region rects are disjoint, so a line
    drawn once per rect touches every exposed pixel exactly once.
    """
    engine = p.paintEngine()
    if engine is None or engine.type() != _RASTER or p.hasClipping():
        return None
    system = engine.systemClip()
    if system.isEmpty():
        return None
    near = p.deviceTransform().mapRect(bounds).toAlignedRect().adjusted(-2, -2, 2, 2)
    world = p.worldTransform()
    p.setWorldTransform(QTransform())
    base = p.deviceTransform()  # redirection offset (e.g. QWidget.grab) and pixel ratio
    p.setWorldTransform(world)
    scale, dx, dy = base.m11(), base.dx(), base.dy()
    # Power-of-two ratios map device rects to logical rects and back exactly.
    if (base.type() not in _SCALED or base.m22() != scale or scale not in (1., 2., 4.)
            or not (dx.is_integer() and dy.is_integer())):
        return None
    rects = [rect for rect in system if rect.intersects(near)]
    if len(rects) > limit:
        return None
    if scale == 1.:
        return [rect.translated(-int(dx), -int(dy)) for rect in rects]
    return [QRectF((r.x() - dx) / scale, (r.y() - dy) / scale, r.width() / scale, r.height() / scale)
            for r in rects]


class GridAxis(pg.AxisItem):
    """AxisItem whose tick and grid lines replay under an explicit exposed clip.

    The recorded picture is split in two: lines (as AxisItem.drawPicture draws
    them) and text in the base picture's state. Both replay unchanged; only the
    clip representation of the identical exposed pixels differs. Qt or pyqtgraph
    versions other than those qualified keep AxisItem's own label and paint.
    """
    _lines = None
    _label_state = None

    def _updateLabel(self):  # noqa: N802
        if not qualified("Qt", "PySide6", "pyqtgraph"):
            return super()._updateLabel()
        # Every range change re-runs this; only re-parse label HTML that changed.
        html = self.labelString()
        document = self.label.document()
        if self._label_state != (html, document, document.revision()):
            self.label.setHtml(html)
            self._label_state = (html, document, document.revision())
        self._adjustSize()
        self.picture = None
        self.update()

    def paint(self, p, opt, widget):
        if not qualified("Qt", "PySide6", "pyqtgraph"):
            return super().paint(p, opt, widget)
        if self.picture is None:
            self._record()
        if self._lines is not None:
            rects = _exposed_rects(p, self.boundingRect())
            if rects is None:
                # QPicture.play() leaves the painter transform changed; isolate it from the text.
                p.save()
                self._lines.play(p)
                p.restore()
            for rect in rects or ():
                world = p.worldTransform()
                p.save()
                p.setWorldTransform(QTransform())
                p.setClipRect(rect)
                p.setWorldTransform(world)
                self._lines.play(p)
                p.restore()
        self.picture.play(p)

    def _record(self):
        text = QPicture()
        self._lines = None
        painter = QPainter(text)
        try:
            if self.style["tickFont"]:
                painter.setFont(self.style["tickFont"])
            specs = self.generateDrawSpecs(painter)
            if specs is not None:
                axis, ticks, texts = specs
                lines = QPicture()
                line_painter = QPainter(lines)
                try:
                    self.drawPicture(line_painter, axis, ticks, [])
                finally:
                    line_painter.end()
                self._lines = lines
                self.drawPicture(painter, (QPen(Qt.PenStyle.NoPen), axis[1], axis[2]), [], texts)
        finally:
            painter.end()
        self.picture = text
