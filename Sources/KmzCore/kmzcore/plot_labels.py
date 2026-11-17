"""Pixel-aware m/z label placement shared by both spectrum widgets."""
import pyqtgraph as pg

from kmzcore.labels import format_mz_label, label_aware_y, select_peak_labels
from kmzcore.theme import INK


def add_peak_labels(plot, mz, intensity, centroid, font):
    view = plot.getViewBox()
    x0, x1 = map(float, view.viewRange()[0])
    width, height = float(view.width()), float(view.height())
    chosen = select_peak_labels(mz, intensity, x0, x1, centroid=centroid, view_width_px=width)
    items = []
    for mass, peak in chosen:
        item = pg.TextItem(format_mz_label(mass), color=INK, anchor=(.5, 1.15))
        item.setFont(font)
        rect = item.textItem.boundingRect()
        if height <= rect.height() * 1.15 + 6 or width < rect.width() + 6:
            continue  # A collapsed pane cannot contain a legible label.
        # Change the anchor, not the measured m/z position or the user's X window.
        px = (mass - x0) / (x1 - x0) * width
        half = rect.width() / 2
        shift = max(3 - (px - half), 0) - max(px + half - (width - 3), 0)
        item.setAnchor((.5 - shift / rect.width(), 1.15))
        item.setPos(mass, peak)
        item.setZValue(20)
        plot.addItem(item, ignoreBounds=True)
        items.append(item)
    if items:
        top_px = max(item.textItem.boundingRect().height() * 1.15 for item in items) + 3
        anchors = [(item.pos().x(), item.pos().y()) for item in items]
        y0, y1 = map(float, view.viewRange()[1])
        plot.setYRange(*label_aware_y(y0, y1, anchors, height, top_px), padding=0)
    return items
