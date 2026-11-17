"""Small accessible eye buttons for file-level MS1 inclusion."""
from functools import lru_cache
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QToolButton


@lru_cache(maxsize=2)
def eye_icon(visible):
    pixmap = QPixmap(40, 40)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#265c51" if visible else "#74818d"), 1.5))
    path = QPainterPath()
    path.moveTo(2, 10)
    path.cubicTo(6, 3, 14, 3, 18, 10)
    path.cubicTo(14, 17, 6, 17, 2, 10)
    painter.drawPath(path)
    painter.setBrush(QColor("#265c51" if visible else "#74818d"))
    painter.drawEllipse(7, 7, 6, 6)
    if not visible:
        painter.setPen(QPen(QColor("#ffffff"), 3.5))
        painter.drawLine(3, 17, 17, 3)
        painter.setPen(QPen(QColor("#74818d"), 1.5))
        painter.drawLine(3, 17, 17, 3)
    painter.end()
    return QIcon(pixmap)


class SpectrumEye(QToolButton):
    def __init__(self, name, checked, callback, parent=None, *, detector='MS1'):
        super().__init__(parent)
        self.name = name
        self.detector = detector
        self._shown_visible = None
        self.setCheckable(True)
        self.setFixedSize(28, 26)
        self.setStyleSheet("QToolButton { padding: 2px; border: 0; }")
        self.setAutoRaise(True)
        self.set_visible(checked)
        self.clicked.connect(callback)

    def set_visible(self, visible):
        self.setChecked(visible)
        if visible == self._shown_visible:
            return
        self._shown_visible = visible
        self.setIcon(eye_icon(visible))
        text = f"{'Hide' if visible else 'Show'} {self.name} in {self.detector} spectra"
        self.setToolTip(text)
        self.setAccessibleName(text)
