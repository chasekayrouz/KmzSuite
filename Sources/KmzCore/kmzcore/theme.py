"""Shared light presentation for the Qt viewers (independent of OS appearance)."""
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleFactory

BACKGROUND = "#f4f6f8"
PAPER = "#ffffff"
INK = "#243349"
MUTED = "#526477"
BORDER = "#ccd6de"
ACCENT = "#265c51"
SELECTION = "#dceee7"
SPECTRUM = "#0072b2"


class VisibleIndicators(QProxyStyle):
    """Explicit outlines/glyphs for widgets AND item-view check indicators."""
    def drawPrimitive(self, element, option, painter, widget=None):
        indicators = (QStyle.PrimitiveElement.PE_IndicatorCheckBox,
                      QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
                      QStyle.PrimitiveElement.PE_IndicatorRadioButton)
        if element not in indicators:
            return super().drawPrimitive(element, option, painter, widget)
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        selected = bool(option.state & (QStyle.StateFlag.State_On | QStyle.StateFlag.State_NoChange))
        focused = bool(option.state & (QStyle.StateFlag.State_HasFocus | QStyle.StateFlag.State_MouseOver))
        ink = QColor(ACCENT if enabled else "#74818d")
        outline = QColor(ACCENT if enabled and focused else "#647687" if enabled else "#8a98a5")
        rect = option.rect.toRectF().adjusted(1, 1, -1, -1)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(outline, 1.2))
        painter.setBrush(QColor(SELECTION if selected and enabled else PAPER if enabled else "#edf0f3"))
        if element == QStyle.PrimitiveElement.PE_IndicatorRadioButton:
            painter.drawEllipse(rect)
            if selected:
                painter.setBrush(ink)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(rect.adjusted(3, 3, -3, -3))
        else:
            painter.drawRoundedRect(rect, 2, 2)
            pen = QPen(ink, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            if option.state & QStyle.StateFlag.State_NoChange:
                painter.drawLine(rect.left() + 3, rect.center().y(), rect.right() - 3, rect.center().y())
            elif selected:
                path = QPainterPath()
                path.moveTo(rect.left() + rect.width() * .20, rect.top() + rect.height() * .52)
                path.lineTo(rect.left() + rect.width() * .43, rect.top() + rect.height() * .75)
                path.lineTo(rect.left() + rect.width() * .82, rect.top() + rect.height() * .25)
                painter.drawPath(path)
        painter.restore()


def apply_theme(app):
    if app is None or app.property("kmzThemeApplied"):
        return
    # QMacStyle's tree check indicators can disappear in native dark appearance.
    # Fusion plus a complete palette keeps widgets and item delegates consistent.
    app._kmz_indicator_style = VisibleIndicators(QStyleFactory.create("Fusion"))
    app.setStyle(app._kmz_indicator_style)
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: INK,
        QPalette.ColorRole.Base: PAPER,
        QPalette.ColorRole.AlternateBase: "#eef3f4",
        QPalette.ColorRole.Text: INK,
        QPalette.ColorRole.Button: PAPER,
        QPalette.ColorRole.ButtonText: INK,
        QPalette.ColorRole.Highlight: SELECTION,
        QPalette.ColorRole.HighlightedText: INK,
        QPalette.ColorRole.ToolTipBase: PAPER,
        QPalette.ColorRole.ToolTipText: INK,
        QPalette.ColorRole.PlaceholderText: MUTED,
        QPalette.ColorRole.Light: PAPER,
        QPalette.ColorRole.Midlight: "#e4eaf0",
        QPalette.ColorRole.Mid: "#9aaab7",
        QPalette.ColorRole.Dark: "#647687",
        QPalette.ColorRole.Shadow: INK,
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.Accent: ACCENT,
    }
    for role, value in roles.items():
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
            palette.setColor(group, role, QColor(value))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#74818d"))
    for role in (QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#edf0f3"))
    app.setPalette(palette)
    font = QFont("Helvetica" if sys.platform == "darwin" else app.font().family(), 11)
    app.setFont(font)
    app.setStyleSheet(f"""
        QToolBar {{ background: {PAPER}; border: 0; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 5px; }}
        QPushButton, QToolButton {{ border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 9px; background: {PAPER}; color: {INK}; }}
        QPushButton:hover, QToolButton:hover {{ background: #eef5f1; border-color: #6e998b; }}
        QPushButton:focus, QToolButton:focus {{ border: 1px solid {ACCENT}; }}
        QPushButton:pressed, QToolButton:pressed, QToolButton:checked {{ background: {SELECTION}; border-color: {ACCENT}; color: {ACCENT}; }}
        QPushButton:disabled, QToolButton:disabled {{ background: #edf0f3; color: #74818d; border-color: #d5dce2; }}
        QLineEdit, QAbstractSpinBox, QComboBox {{ background: {PAPER}; color: {INK}; border: 1px solid #aebdc9; border-radius: 5px; padding: 4px; selection-background-color: {SELECTION}; selection-color: {INK}; }}
        QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
        QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ background: #edf0f3; color: #74818d; border-color: #d5dce2; }}
        QAbstractItemView {{ background: {PAPER}; color: {INK}; border: 1px solid {BORDER}; selection-background-color: {SELECTION}; selection-color: {INK}; outline: 0; }}
        QAbstractItemView::item {{ padding: 4px; }}
        QAbstractItemView::item:hover {{ background: #eef5f1; }}
        QAbstractItemView::item:selected {{ background: {SELECTION}; color: {INK}; }}
        QAbstractItemView::item:focus {{ border: 1px solid {ACCENT}; }}
        QHeaderView::section {{ background: #eef3f4; color: {INK}; padding: 5px; border: 0; border-bottom: 1px solid {BORDER}; }}
        QDockWidget::title {{ background: #eaf0f2; color: {INK}; padding: 5px; }}
        QStatusBar {{ color: {MUTED}; }}
        QCheckBox, QRadioButton {{ spacing: 6px; }}
        QCheckBox:disabled, QRadioButton:disabled {{ color: #74818d; }}
        QCheckBox:focus, QRadioButton:focus {{ color: {ACCENT}; }}
        QMenu {{ background: {PAPER}; color: {INK}; border: 1px solid {BORDER}; }}
        QMenu::item:selected {{ background: {SELECTION}; color: {INK}; }}
        QMenu::item:disabled {{ color: #74818d; }}
    """)
    app.setProperty("kmzThemeApplied", True)


def configure_plots():
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background=PAPER, foreground=INK)


def color_icon(color):
    from PySide6.QtGui import QIcon, QPainter, QPen, QPixmap
    swatch = QPixmap(18, 18)
    swatch.fill(Qt.GlobalColor.transparent)
    painter = QPainter(swatch)
    painter.setPen(QPen(QColor(BORDER)))
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(1, 3, 15, 12, 2, 2)
    painter.end()
    return QIcon(swatch)
