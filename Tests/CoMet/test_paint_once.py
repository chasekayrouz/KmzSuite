"""EIC tabs paint their dense traces once per interaction, at the final geometry."""
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication

from kmzcomet.chromatograms import EICBatch, EICData, EICEntry
from kmzcomet.model import Feature, Measurement, Result, Sample, Settings
from kmzcomet_app.app import MainWindow
from kmzcomet_app.workers import Completed


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


class Paints(QObject):
    def __init__(self):
        super().__init__()
        self.count = 0

    def eventFilter(self, _obj, event):  # noqa: N802
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def _result(samples=4):
    runs = [Sample(Path(f'synthetic-{i}.wiff'), f'S{i+1:03d}', '0'*64, 0, units={'counts'}) for i in range(samples)]
    features = [Feature(f'F{i+1:06d}', 200+i, 1+i*.1, 1, [], 0, 6) for i in range(20)]
    columns = [[Measurement(f.id, s.id, True, 'detected', height=float(100+i+j)) for i, f in enumerate(features)]
               for j, s in enumerate(runs)]
    return Result(Settings(), runs, features, columns, 0, analysis_id='paint-test', created_utc='2026-01-01T00:00:00+00:00')


def _batch(count, points=1500):
    rng = np.random.default_rng(3)
    rt = np.linspace(0., 12., points)
    entries = [EICEntry(f'F{i//4+1:06d}', f'S{i%4+1:03d}', f'sample-{i%4}.wiff', None, 200.+i, 1, 30., 'counts',
                        EICData(rt, np.abs(rng.normal(0, 50, points)) + 4000*np.exp(-((rt-1-i*.4)/.05)**2)*(i+1)))
               for i in range(count)]
    return EICBatch(tuple(entries), 0, 4)


def test_eic_tab_interactions_paint_once(qapp, tmp_path):
    window = MainWindow(history=tmp_path/'history')
    window.resize(1300, 860); window.show()
    try:
        window._completed(Completed(_result(), tmp_path/'saved.kmzcomet'))
        window._eics_ready(_batch(8))
        tab = window.tabs.currentWidget()
        paints = Paints(); tab.plots.view.viewport().installEventFilter(paints)

        def painted(action):
            for _ in range(3):
                qapp.processEvents()
            paints.count = 0
            action()
            for _ in range(6):
                qapp.processEvents()
            return paints.count

        item = tab.legend.item(1)
        # Each relayout (stack height, legend splitter, dock, window) or range
        # change repaints the dense traces exactly once, at the final geometry.
        for action, expected in ((tab.overlay_action.trigger, 1), (lambda: tab.plots.set_x_range(2., 5.), 1),
                                 (lambda: tab.toggle_emphasis(2), 1), (tab.stacked_action.trigger, 1),
                                 (lambda: item.setCheckState(Qt.CheckState.Unchecked), 1),
                                 (lambda: item.setCheckState(Qt.CheckState.Checked), 1),
                                 (tab.legend_action.toggle, 1), (tab.legend_action.toggle, 1),
                                 (lambda: window.tabs.setCurrentIndex(0), 0), (lambda: window.tabs.setCurrentWidget(tab), 1),
                                 (lambda: window.resize(1200, 800), 1)):
            assert painted(action) == expected
        assert not window.dock.isVisible()
    finally:
        window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); qapp.processEvents()


def test_new_eic_tab_paints_once(qapp, tmp_path):
    window = MainWindow(history=tmp_path/'history')
    window.resize(1300, 860); window.show()
    try:
        window._completed(Completed(_result(), tmp_path/'saved.kmzcomet'))
        for _ in range(3):
            qapp.processEvents()
        assert window.dock.isVisible()
        views = []

        class ViewPaints(QObject):
            def eventFilter(self, obj, event):  # noqa: N802
                if event.type() == QEvent.Type.Paint and type(obj.parent()).__name__ == 'GraphicsLayoutWidget':
                    views.append(obj.width())
                return False
        counter = ViewPaints(); qapp.installEventFilter(counter)
        try:
            window._eics_ready(_batch(8))
            for _ in range(6):
                qapp.processEvents()
        finally:
            qapp.removeEventFilter(counter)
        tab = window.tabs.currentWidget()
        assert not window.dock.isVisible()
        assert views == [tab.plots.view.viewport().width()]
    finally:
        window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); qapp.processEvents()


def test_heights_identifier_cells_paint_once(qapp, tmp_path, monkeypatch):
    from collections import Counter
    from kmzcomet_app.table import FeatureFilter
    requested = Counter()
    original = FeatureFilter.data

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            requested[index.row(), index.column()] += 1
        return original(self, index, role)
    monkeypatch.setattr(FeatureFilter, 'data', data)
    window = MainWindow(history=tmp_path/'history')
    window.resize(1300, 860); window.show()
    try:
        window._completed(Completed(_result(), tmp_path/'saved.kmzcomet'))
        for _ in range(3):
            qapp.processEvents()
        requested.clear()
        window.table.viewport().update(); window.table.frozen.viewport().update()
        for _ in range(3):
            qapp.processEvents()
        # The opaque frozen view hides the main view's identifying columns.
        frozen = [count for (row, column), count in requested.items() if column < 5]
        samples = [count for (row, column), count in requested.items() if column >= 5]
        assert frozen and samples and set(frozen) == {1} and set(samples) == {1}
        rows = {row for row, column in requested if column >= 5}
        assert set(requested) >= {(row, column) for row in rows for column in range(5)}
    finally:
        window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); qapp.processEvents()
