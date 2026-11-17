"""Independent review regressions. Baseline proxy is the oracle; no patch fixes here."""
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QItemSelectionModel, QModelIndex, QPersistentModelIndex, Qt
from PySide6.QtWidgets import QApplication

from kmzcomet.model import Feature, Measurement, Result, Sample, Settings
from kmzcomet_app.table import FeatureFilter, MatrixModel
from Tests.CoMet.test_feature_filter import QtFeatureFilter


@pytest.fixture(scope='module')
def application():
    return QApplication.instance() or QApplication([])


def result(values):
    samples = [Sample(Path('synthetic.mzML'), 'S', '0'*64, 0)]
    features = [Feature(f'F{i}', 100.+i, i*.1, 1, []) for i in range(len(values))]
    measurements = [[Measurement(f.id, 'S', True, 'detected', height=v) for f, v in zip(features, values)]]
    return Result(Settings(), samples, features, measurements, 0)


@pytest.mark.parametrize('proxy_type', [QtFeatureFilter, FeatureFilter])
@pytest.mark.parametrize('reset', [False, True])
def test_map_from_source_initializes_its_lazy_mapping(application, proxy_type, reset):
    model = MatrixModel(None if reset else result([3., 2., 1.]))
    proxy = proxy_type(); proxy.setSourceModel(model)
    if reset:
        model.set_result(result([3., 2., 1.]))
    # This is a valid first query; no view or rowCount call is required by Qt.
    mapped = proxy.mapFromSource(model.index(1, 0))
    assert mapped.isValid() and mapped.row() == 1 and mapped.data() == 'F1'


@pytest.mark.parametrize('proxy_type', [QtFeatureFilter, FeatureFilter])
@pytest.mark.parametrize('action', ['insert', 'remove', 'layout'])
def test_source_structure_preserves_surviving_selection(application, proxy_type, action):
    model = MatrixModel(result([3., 2., 1.]))
    proxy = proxy_type(); proxy.setSourceModel(model); proxy.sort(4, Qt.SortOrder.AscendingOrder)
    selection = QItemSelectionModel(proxy)
    selection.setCurrentIndex(proxy.index(1, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    persistent = QPersistentModelIndex(proxy.index(1, 0))
    signals = []
    for name in ('modelReset', 'rowsInserted', 'rowsRemoved', 'layoutChanged'):
        getattr(proxy, name).connect(lambda *_, name=name: signals.append(name))
    if action == 'insert':
        model.beginInsertRows(QModelIndex(), 0, 0)
        model.result.features.insert(0, Feature('NEW', 50., 0., 1, []))
        model.result.measurements[0].insert(0, Measurement('NEW', 'S', True, 'detected', height=0.))
        model.endInsertRows()
    elif action == 'remove':
        model.beginRemoveRows(QModelIndex(), 0, 0)
        model.result.features.pop(0); model.result.measurements[0].pop(0)
        model.endRemoveRows()
    else:
        model.layoutAboutToBeChanged.emit(); model.layoutChanged.emit()
    assert [i.data() for i in selection.selectedIndexes()] == ['F1']
    assert selection.currentIndex().data() == 'F1'
    assert persistent.isValid() and persistent.data() == 'F1'
    assert signals == [{'insert': 'rowsInserted', 'remove': 'rowsRemoved', 'layout': 'layoutChanged'}[action]]


@pytest.mark.parametrize('proxy_type', [QtFeatureFilter, FeatureFilter])
def test_source_destruction_clears_cached_mapping(application, proxy_type):
    model = MatrixModel(result([3., 2., 1.]))
    proxy = proxy_type(); proxy.setSourceModel(model)
    assert proxy.rowCount() == 3
    model.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert proxy.rowCount() == 0
    assert not proxy.index(0, 0).isValid()


@pytest.mark.parametrize('proxy_type', [QtFeatureFilter, FeatureFilter])
def test_new_source_can_follow_a_destroyed_source(application, proxy_type):
    model = MatrixModel(result([3., 2., 1.]))
    proxy = proxy_type(); proxy.setSourceModel(model)
    assert proxy.rowCount() == 3
    model.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    replacement = MatrixModel(result([4.]))
    proxy.setSourceModel(replacement)
    assert proxy.rowCount() == 1
