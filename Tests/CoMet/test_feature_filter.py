"""The Heights proxy must reproduce Qt's QSortFilterProxyModel row mapping exactly."""
import math
import random
from pathlib import Path

import pytest
from PySide6.QtCore import (QCoreApplication, QEvent, QItemSelectionModel, QModelIndex, QPersistentModelIndex,
    QSortFilterProxyModel, Qt)
from PySide6.QtWidgets import QApplication, QTableView

from kmzcomet.model import Feature, Measurement, Result, Sample, Settings
from kmzcomet_app.table import LAYOUT, ROLE, EICMatrixModel, FeatureFilter, MatrixModel, feature_filter, libcxx_stable_sort
from kmzcomet_app import table

ASC, DESC = Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder


class QtFeatureFilter(QSortFilterProxyModel):
    """The previous Heights proxy: Qt's own mapping with per-comparison data() calls."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bounds = (None, None, None, None)
        self.text = ''
        self.setSortRole(ROLE)

    def set_filters(self, text, bounds):
        self.beginFilterChange()
        self.text, self.bounds = text.strip().lower(), bounds
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        if model.result is None:
            return False
        f = model.result.features[row]
        lo, hi, start, end = self.bounds
        if ((lo is not None and f.mz < lo) or (hi is not None and f.mz > hi)
                or (start is not None and f.rt < start) or (end is not None and f.rt > end)):
            return False
        return not self.text or self.text in f'{f.id} {f.mz:.6f} {f.rt:.4f}'.lower()

    def lessThan(self, left, right):
        a, b = left.data(ROLE), right.data(ROLE)
        if a is None or b is None:
            if a is b:
                return left.row() < right.row()
            return (a is not None) if self.sortOrder() == Qt.SortOrder.AscendingOrder else (a is None)
        return a < b


@pytest.fixture(scope='module')
def qt_application():
    return QApplication.instance() or QApplication([])


def random_result(rng, *, nan=False):
    count, samples = rng.randrange(0, 60), rng.randrange(1, 4)
    runs = [Sample(Path(f's{i}.mzML'), f'S{i}', '0'*64, 0) for i in range(samples)]
    features = [Feature(f'F{rng.randrange(40):03d}', 100+rng.randrange(8)*.5, rng.randrange(6)*.25,
                        rng.choice((1, 1, -1, 0)), []) for _ in range(count)]
    values = [None, 0., -0., 1., 2., 2., 7.5, math.inf] + ([math.nan] if nan else [])
    columns = [[Measurement(f.id, s.id, True, 'detected', height=rng.choice(values)) for f in features] for s in runs]
    return Result(Settings(), runs, features, columns, 0)


def random_filter(rng):
    bounds = [None]*4
    for i in rng.sample(range(4), rng.randrange(3)):
        bounds[i] = (100+rng.randrange(8)*.5) if i < 2 else rng.randrange(6)*.25
    return rng.choice(('', '', 'f00', '101', '0.5', 'F01')), tuple(bounds)


class Pair:
    """Drive the Qt reference and the port identically; record what views would see."""
    def __init__(self, rng, eic, views):
        self.models, self.proxies, self.selections, self.logs, self.persistent = [], [], [], [], [[], []]
        for cls in (QtFeatureFilter, FeatureFilter):
            model = EICMatrixModel() if eic else MatrixModel()
            proxy = cls(); proxy.setSourceModel(model)
            log = []
            for name in ('rowsAboutToBeInserted', 'rowsInserted', 'rowsAboutToBeRemoved', 'rowsRemoved',
                         'columnsAboutToBeInserted', 'columnsInserted', 'columnsAboutToBeRemoved', 'columnsRemoved',
                         'modelAboutToBeReset', 'modelReset', 'dataChanged', 'headerDataChanged'):
                getattr(proxy, name).connect(lambda *a, name=name, log=log: log.append((name, describe(a))))
            for name in ('layoutAboutToBeChanged', 'layoutChanged'):
                getattr(proxy, name)[LAYOUT].connect(lambda *a, name=name, log=log: log.append((name, describe(a))))
            if views:
                view = QTableView(); view.setModel(proxy); view.setSortingEnabled(False)
                proxy.view = view
                selection = view.selectionModel()
            else:
                selection = QItemSelectionModel(proxy)
            self.models.append(model); self.proxies.append(proxy); self.selections.append(selection); self.logs.append(log)

    def state(self, proxy, selection, persistent):
        order = [proxy.mapToSource(proxy.index(i, 0)).row() for i in range(proxy.rowCount())]
        source = lambda index: (proxy.mapToSource(index).row(), index.column())
        columns = range(-1, proxy.columnCount()+1)
        cells = [[proxy.index(row, column).data() for column in columns] for row in range(min(3, proxy.rowCount()))]
        headers = [proxy.headerData(column, Qt.Orientation.Horizontal) for column in columns]
        kept = [(index.isValid(), source(proxy.index(index.row(), index.column()))) for index in persistent]
        return (order, proxy.rowCount(), proxy.columnCount(), proxy.sortColumn(), proxy.sortOrder(),
                sorted(source(i) for i in selection.selectedIndexes()), source(selection.currentIndex()),
                cells, headers, kept)

    def check(self):
        states = [self.state(*args) for args in zip(self.proxies, self.selections, self.persistent)]
        assert states[0] == states[1]
        assert self.logs[0] == self.logs[1]
        for log in self.logs:
            log.clear()

    def select(self, rng):
        proxy = self.proxies[0]
        if not proxy.rowCount():
            return
        cells = [(rng.randrange(proxy.rowCount()), rng.randrange(proxy.columnCount())) for _ in range(rng.randrange(4))]
        current = (rng.randrange(proxy.rowCount()), rng.randrange(proxy.columnCount()))
        for p, s, persistent in zip(self.proxies, self.selections, self.persistent):
            s.clearSelection()
            for row, column in cells:
                s.select(p.index(row, column), QItemSelectionModel.SelectionFlag.Select)
            s.setCurrentIndex(p.index(*current), QItemSelectionModel.SelectionFlag.NoUpdate)
            persistent[:] = [QPersistentModelIndex(p.index(row, column)) for row, column in cells[:2]]

    def apply(self, action):
        results = [action(model, proxy) for model, proxy in zip(self.models, self.proxies)]
        assert results[0] == results[1]


def describe(args):
    cell = lambda a: (a.row(), a.column()) if hasattr(a, 'row') else a
    return tuple([cell(i) for i in a] if isinstance(a, list) else cell(a) for a in args)


@pytest.mark.parametrize('seed', range(24))
def test_port_matches_qt_proxy_mapping_signals_and_selection(qt_application, seed):
    rng = random.Random(seed)
    eic, views, nan = seed % 2 == 0, seed % 3 != 0, seed % 4 == 3
    pair = Pair(rng, eic, views)
    for step in range(120):
        roll = rng.random()
        if roll < .1 or step == 0:
            result = random_result(rng, nan=nan)
            pair.apply(lambda model, proxy: model.set_result(result))
        elif roll < .45:
            column = rng.randrange(-1, (pair.models[0].columnCount() if rng.random() < .9 else 30))
            order = rng.choice((ASC, DESC))
            pair.apply(lambda model, proxy: proxy.sort(column, order))
        elif roll < .7:
            text, bounds = random_filter(rng)
            pair.apply(lambda model, proxy: proxy.set_filters(text, bounds))
        elif roll < .9 and eic and pair.models[0].rowCount():
            row = rng.randrange(pair.models[0].rowCount()); sample = rng.randrange(len(pair.models[0].result.samples))
            pair.apply(lambda model, proxy: model.toggle_cell(row, sample))
        elif roll < .95 and eic:
            pair.apply(lambda model, proxy: model.clear_picks())
        else:
            pair.select(rng)
        pair.check()


def own(result):
    """A copy whose lists each source model may edit in place."""
    return Result(result.settings, list(result.samples), list(result.features), [list(c) for c in result.measurements], 0)


def remap_picks(model, rows=lambda row: row, samples=lambda sample: sample):
    if isinstance(model, EICMatrixModel):
        picks = {rows(row): {samples(s) for s in chosen} - {None} for row, chosen in model.picks.items()}
        model.picks.clear(); model.picks.update({row: chosen for row, chosen in picks.items() if row is not None and chosen})


def permute_rows(model, order):
    """Row p takes old row order[p]; persistent source indexes follow their rows."""
    result, position = model.result, {old: new for new, old in enumerate(order)}
    result.features[:] = [result.features[i] for i in order]
    for column in result.measurements:
        column[:] = [column[i] for i in order]
    remap_picks(model, rows=position.get)
    return position


def moved(count, first, last, destination):
    rest, block = [i for i in range(count) if not first <= i <= last], list(range(first, last+1))
    at = destination if destination < first else destination-len(block)
    return rest[:at] + block + rest[at:]


def structure_change(rng, model, nan):
    """One random source edit as an action for Pair.apply()."""
    rows, samples, base = model.rowCount(), len(model.result.samples), model.columnCount()-len(model.result.samples)
    values = [None, 0., 1., 2., 7.5, math.inf] + ([math.nan] if nan else [])
    kind = rng.choice(('insert', 'remove', 'move', 'layout', 'insert columns', 'remove columns', 'move columns', 'header'))
    if kind == 'insert':
        start, count = rng.randrange(rows+1), rng.randrange(1, 4)
        new = [(f'N{rng.randrange(40):03d}', 100+rng.randrange(8)*.5, rng.randrange(6)*.25,
                [rng.choice(values) for _ in range(samples)]) for _ in range(count)]
        def action(model, proxy):
            result = model.result
            model.beginInsertRows(QModelIndex(), start, start+count-1)
            result.features[start:start] = [Feature(i, mz, rt, 1, []) for i, mz, rt, _ in new]
            for s, column in enumerate(result.measurements):
                column[start:start] = [Measurement(i, result.samples[s].id, True, 'detected', height=h[s]) for i, _, _, h in new]
            remap_picks(model, rows=lambda row: row+count if row >= start else row)
            model.endInsertRows()
    elif kind == 'remove' and rows:
        start = rng.randrange(rows); end = min(rows-1, start+rng.randrange(3))
        def action(model, proxy):
            model.beginRemoveRows(QModelIndex(), start, end)
            del model.result.features[start:end+1]
            for column in model.result.measurements:
                del column[start:end+1]
            remap_picks(model, rows=lambda row: None if start <= row <= end else row-(end-start+1) if row > end else row)
            model.endRemoveRows()
    elif kind == 'move' and rows > 1:
        first = rng.randrange(rows); last = min(rows-1, first+rng.randrange(3))
        destination = rng.choice([i for i in range(rows+1) if not first <= i <= last+1] or [None])
        if destination is None:
            return None
        def action(model, proxy):
            assert model.beginMoveRows(QModelIndex(), first, last, QModelIndex(), destination)
            permute_rows(model, moved(rows, first, last, destination))
            model.endMoveRows()
    elif kind == 'layout':
        order = list(range(rows))
        if rng.random() < .7:
            rng.shuffle(order)
        explicit = rng.random() < .5
        def action(model, proxy):
            if explicit:
                model.layoutAboutToBeChanged[LAYOUT].emit([QPersistentModelIndex()], QSortFilterProxyModel.LayoutChangeHint.VerticalSortHint)
            else:
                model.layoutAboutToBeChanged.emit()
            position = permute_rows(model, order)
            old = model.persistentIndexList()
            model.changePersistentIndexList(old, [model.index(position[i.row()], i.column()) for i in old])
            if explicit:
                model.layoutChanged[LAYOUT].emit([QPersistentModelIndex()], QSortFilterProxyModel.LayoutChangeHint.VerticalSortHint)
            else:
                model.layoutChanged.emit()
    elif kind == 'insert columns':
        start, count = rng.randrange(samples+1), rng.randrange(1, 3)
        heights = [[rng.choice(values) for _ in range(rows)] for _ in range(count)]
        names = [f'X{rng.randrange(1000)}' for _ in range(count)]
        def action(model, proxy):
            result = model.result
            model.beginInsertColumns(QModelIndex(), base+start, base+start+count-1)
            result.samples[start:start] = new = [Sample(Path(f'{n}.mzML'), n, '0'*64, 0) for n in names]
            result.measurements[start:start] = [[Measurement(f.id, s.id, True, 'detected', height=h) for f, h in zip(result.features, column)]
                                                for s, column in zip(new, heights)]
            remap_picks(model, samples=lambda s: s+count if s >= start else s)
            model.endInsertColumns()
    elif kind == 'remove columns' and samples:
        start = rng.randrange(samples); end = min(samples-1, start+rng.randrange(2))
        def action(model, proxy):
            model.beginRemoveColumns(QModelIndex(), base+start, base+end)
            del model.result.samples[start:end+1], model.result.measurements[start:end+1]
            remap_picks(model, samples=lambda s: None if start <= s <= end else s-(end-start+1) if s > end else s)
            model.endRemoveColumns()
    elif kind == 'move columns' and samples > 1:
        first = rng.randrange(samples); last = min(samples-1, first+rng.randrange(2))
        destination = rng.choice([i for i in range(samples+1) if not first <= i <= last+1] or [None])
        if destination is None:
            return None
        order = moved(samples, first, last, destination)
        def action(model, proxy):
            assert model.beginMoveColumns(QModelIndex(), base+first, base+last, QModelIndex(), base+destination)
            result, position = model.result, {old: new for new, old in enumerate(order)}
            result.samples[:] = [result.samples[i] for i in order]
            result.measurements[:] = [result.measurements[i] for i in order]
            remap_picks(model, samples=position.get)
            model.endMoveColumns()
    elif kind == 'header':
        vertical = rows and rng.random() < .5
        count = rows if vertical else model.columnCount()
        first = rng.randrange(count); last = rng.randrange(first, count)
        orientation = Qt.Orientation.Vertical if vertical else Qt.Orientation.Horizontal
        def action(model, proxy):
            model.headerDataChanged.emit(orientation, first, last)
    else:
        return None
    return action


@pytest.mark.parametrize('seed', range(24))
def test_port_matches_qt_proxy_under_source_structure_changes(qt_application, seed):
    rng = random.Random(1000+seed)
    eic, views, nan = seed % 2 == 0, seed % 3 != 0, seed % 4 == 3
    pair = Pair(rng, eic, views)
    for step in range(150):
        roll = rng.random()
        if roll < .08 or step == 0:
            result = random_result(rng, nan=nan)
            # Unmapped proxies must answer first queries and source changes as Qt does.
            source = (rng.randrange(-1, 4), rng.randrange(-1, 8))
            follow = rng.choice(('map', 'sort', 'toggle', 'header', 'none'))
            def action(model, proxy):
                model.set_result(own(result))
                if follow == 'map':
                    return describe([proxy.mapFromSource(model.index(*source))])
                if follow == 'sort':
                    proxy.sort(4, DESC)
                elif follow == 'toggle' and eic and model.rowCount():
                    model.toggle_cell(0, 0)
                elif follow == 'header' and model.columnCount():
                    model.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, model.columnCount()-1)
            pair.apply(action)
        elif roll < .45:
            action = structure_change(rng, pair.models[0], nan)
            if action is not None:
                pair.apply(action)
        elif roll < .6:
            column = rng.randrange(-1, pair.models[0].columnCount()+1)
            order = rng.choice((ASC, DESC))
            pair.apply(lambda model, proxy: proxy.sort(column, order))
        elif roll < .72:
            text, bounds = random_filter(rng)
            pair.apply(lambda model, proxy: proxy.set_filters(text, bounds))
        elif roll < .8 and eic and pair.models[0].rowCount() and pair.models[0].result.samples:
            row = rng.randrange(pair.models[0].rowCount()); sample = rng.randrange(len(pair.models[0].result.samples))
            pair.apply(lambda model, proxy: model.toggle_cell(row, sample))
        else:
            pair.select(rng)
        pair.check()


@pytest.mark.parametrize('views', [False, True])
def test_source_destruction_matches_qt(qt_application, capfd, views):
    rng = random.Random(5)
    pair = Pair(rng, True, views)
    result = random_result(rng)
    while len(result.features) < 4:
        result = random_result(rng)
    pair.apply(lambda model, proxy: model.set_result(own(result)))
    pair.apply(lambda model, proxy: proxy.sort(4, DESC)); pair.select(rng); pair.check()
    for model in pair.models:
        model.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    assert [p.sourceModel() for p in pair.proxies] == [None, None]
    pair.check()
    pair.models = [EICMatrixModel(own(result)) for _ in pair.proxies]
    pair.apply(lambda model, proxy: proxy.setSourceModel(model))
    pair.check()
    pair.apply(lambda model, proxy: model.toggle_cell(1, 0)); pair.check()
    pair.apply(lambda model, proxy: proxy.setSourceModel(None)); pair.check()
    assert 'Traceback' not in capfd.readouterr().err


def test_libcxx_stable_sort_is_stable_and_total_for_weak_orders():
    rng = random.Random(7)
    for size in (0, 1, 2, 3, 8, 9, 129, 300, 1000):
        values = [rng.randrange(20) for _ in range(size)]
        assert libcxx_stable_sort(range(size), lambda a, b: values[a] < values[b]) == sorted(range(size), key=values.__getitem__)


def test_large_heights_sort_and_filter_match_qt(qt_application):
    rng = random.Random(3)
    runs = [Sample(Path('a.mzML'), 'A', '0'*64, 0), Sample(Path('b.mzML'), 'B', '0'*64, 0)]
    features = [Feature(f'F{i:06d}', 100+(i % 900)*.1, (i % 60)*.01, 1, []) for i in range(3000)]
    columns = [[Measurement(f.id, s.id, True, 'detected', height=rng.choice((None, 0., float(rng.randrange(50)))))
                for f in features] for s in runs]
    result = Result(Settings(), runs, features, columns, 0)
    pair = Pair(rng, True, True)
    pair.apply(lambda model, proxy: model.set_result(result))
    for column, order in ((2, ASC), (5, DESC), (3, DESC), (6, ASC), (1, DESC)):
        pair.apply(lambda model, proxy: proxy.sort(column, order)); pair.check()
        for bounds in ((150., None, None, None), (None, 120., .2, None), (None, None, None, None)):
            pair.apply(lambda model, proxy: proxy.set_filters('', bounds)); pair.check()


def previous_matrix_data(model, index, role=Qt.ItemDataRole.DisplayRole):
    if not index.isValid() or model.result is None:
        return None
    value = model.value(index.row(), index.column())
    if role == ROLE:
        return value
    if role == Qt.ItemDataRole.TextAlignmentRole:
        return int(Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignRight if isinstance(value, (int,float)) else Qt.AlignmentFlag.AlignLeft))
    if role == Qt.ItemDataRole.DisplayRole:
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.6f}' if index.column() == 1 else f'{value:.4f}' if index.column() == 2 else f'{value:.8g}'
        return str(value)


def previous_eic_data(model, index, role=Qt.ItemDataRole.DisplayRole):
    from PySide6.QtGui import QColor
    if not index.isValid() or model.result is None:
        return None
    row, column = index.row(), index.column()
    if column == 0:
        if role == Qt.ItemDataRole.CheckStateRole:
            count = len(model.picks.get(row, ()))
            return (Qt.CheckState.Checked if count == len(model.result.samples) else
                    Qt.CheckState.PartiallyChecked if count else Qt.CheckState.Unchecked)
        return model.value(row, column) if role == ROLE else None
    picked = column-model.sample_start in model.picks.get(row, ())
    if picked and role == Qt.ItemDataRole.BackgroundRole:
        return QColor('#d5eee3')
    if picked and role == Qt.ItemDataRole.ForegroundRole:
        return QColor('#173d30')
    if role == Qt.ItemDataRole.ToolTipRole and column >= model.sample_start:
        return ('Selected for Plot EICs. Click to deselect.' if picked else 'Click to select this sample EIC.')
    if role == Qt.ItemDataRole.DisplayRole:
        value = model.value(row, column)
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.6f}' if column == 2 else f'{value:.4f}' if column == 3 else f'{value:.8g}'
        return str(value)
    return previous_matrix_data(model, index, role)


def test_cell_paths_match_previous_data_for_every_role(qt_application):
    rng = random.Random(11)
    result = random_result(rng)
    while not result.features:
        result = random_result(rng)
    eic, plain, evidence = EICMatrixModel(), MatrixModel(result), MatrixModel(result, evidence=True)
    eic.set_result(result)
    proxy = FeatureFilter(); proxy.setSourceModel(eic); proxy.sort(3, DESC)
    for _ in range(20):
        eic.toggle_cell(rng.randrange(len(result.features)), rng.randrange(len(result.samples)))
    roles = list(range(16)) + [ROLE]
    for model, previous in ((eic, previous_eic_data), (plain, previous_matrix_data), (evidence, previous_matrix_data)):
        for row in range(model.rowCount()):
            for column in range(model.columnCount()):
                for role in roles:
                    index = model.index(row, column)
                    assert model.data(index, role) == previous(model, index, role)
                    if model is eic:
                        assert proxy.mapFromSource(index).data(role) == previous(model, index, role)


def test_unqualified_qt_uses_qts_own_proxy(qt_application, unqualified):
    rng = random.Random(41)
    result, (text, bounds) = random_result(rng), random_filter(rng)
    assert type(feature_filter()) is FeatureFilter
    unqualified('Qt')
    fallback = feature_filter()
    assert type(fallback) is table.QtFeatureFilter
    models, orders = [], []
    for proxy in (fallback, QtFeatureFilter(), FeatureFilter()):  # Production fallback, frozen original, port.
        models.append(MatrixModel(result))
        proxy.setSourceModel(models[-1])
        proxy.set_filters(text, bounds)
        proxy.sort(4, DESC)
        orders.append([proxy.mapToSource(proxy.index(row, 0)).row() for row in range(proxy.rowCount())])
    assert orders[0] == orders[1] == orders[2]
