"""Virtual result tables with typed sorting; no per-cell widget allocation."""
import csv
from bisect import bisect_left
from io import StringIO
from dataclasses import asdict

from PySide6.QtCore import (QAbstractItemModel, QAbstractProxyModel, QAbstractTableModel, QItemSelectionModel,
    QModelIndex, QObject, QPersistentModelIndex, QSortFilterProxyModel, Qt, Signal)
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import QApplication, QTableView

from kmzcomet.csv_export import sample_label
from kmzcore.qualified import qualified

ROLE = Qt.ItemDataRole.UserRole
DISPLAY, ALIGNMENT, CHECK = Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.TextAlignmentRole, Qt.ItemDataRole.CheckStateRole
BACKGROUND, FOREGROUND, TOOLTIP = Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole, Qt.ItemDataRole.ToolTipRole
NUMBER_ALIGN = int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
TEXT_ALIGN = int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
POLARITY = {1:'positive', -1:'negative', 0:'unknown'}


class MatrixModel(QAbstractTableModel):
    def __init__(self, result=None, *, evidence=False):
        super().__init__()
        self.result = result
        self.evidence = evidence

    def set_result(self, result):
        self.beginResetModel()
        self.result = result
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.result is None else len(self.result.features)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 4+(len(self.result.samples) if self.result else 0)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal or section >= self.columnCount():
            return None
        if section < 4:
            return ('Feature ID', 'm/z', 'RT (min)', 'Polarity')[section] if role == Qt.ItemDataRole.DisplayRole else None
        sample = self.result.samples[section-4]
        if role == Qt.ItemDataRole.DisplayRole:
            return f'{sample.path.stem}\n{sample.id}'
        if role == Qt.ItemDataRole.ToolTipRole:
            return sample_label(sample)

    def value(self, row, column):
        feature = self.result.features[row]
        if column < 4:
            return (feature.id, feature.mz, feature.rt, POLARITY[feature.polarity])[column]
        measurement = self.result.measurements[column-4][row]
        return measurement.status if self.evidence else measurement.height

    def column_values(self, column):
        """value(row, column) for every row, without per-cell index dispatch."""
        if column == 3:
            return [POLARITY[f.polarity] for f in self.result.features]
        if column < 4:
            name = ('id', 'mz', 'rt')[column]
            return [getattr(f, name) for f in self.result.features]
        name = 'status' if self.evidence else 'height'
        return [getattr(m, name) for m in self.result.measurements[column-4]]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        return self.cell(index.row(), index.column(), role) if index.isValid() else None

    def cell(self, row, column, role):
        """data() for a valid cell; the Heights proxy calls it without index objects."""
        if self.result is None or role not in (ROLE, ALIGNMENT, DISPLAY):
            return None
        value = self.value(row, column)
        if role == ROLE:
            return value
        if role == ALIGNMENT:
            return NUMBER_ALIGN if isinstance(value, (int,float)) else TEXT_ALIGN
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.6f}' if column == 1 else f'{value:.4f}' if column == 2 else f'{value:.8g}'
        return str(value)


class EICMatrixModel(MatrixModel):
    """Explicit plotting picks, independent of table focus, sorting and filters."""
    selection_changed = Signal()
    sample_start = 5

    def __init__(self, result=None):
        super().__init__(result)
        self.picks = {}

    @property
    def selected_cells(self):
        return tuple((row, sample) for row in sorted(self.picks) for sample in sorted(self.picks[row]))

    def set_result(self, result):
        self.picks.clear()
        super().set_result(result)
        self.selection_changed.emit()

    def columnCount(self, parent=QModelIndex()):
        return super().columnCount(parent) + (0 if parent.isValid() else 1)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and section == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                return 'Plot'
            if role == Qt.ItemDataRole.ToolTipRole:
                return 'Select every sample EIC in this feature row. A partial check means some samples are selected.'
            return None
        return super().headerData(section-1, orientation, role)

    def value(self, row, column):
        return len(self.picks.get(row, ())) if column == 0 else super().value(row, column-1)

    def column_values(self, column):
        if column:
            return super().column_values(column-1)
        return [len(self.picks.get(row, ())) for row in range(len(self.result.features))]

    def cell(self, row, column, role):
        if self.result is None:
            return None
        if column == 0:
            if role == CHECK:
                count = len(self.picks.get(row, ()))
                return (Qt.CheckState.Checked if count == len(self.result.samples) else
                        Qt.CheckState.PartiallyChecked if count else Qt.CheckState.Unchecked)
            return self.value(row, column) if role == ROLE else None
        if role in (BACKGROUND, FOREGROUND, TOOLTIP):
            picked = column-self.sample_start in self.picks.get(row, ())
            if picked and role == BACKGROUND:
                return QColor('#d5eee3')
            if picked and role == FOREGROUND:
                return QColor('#173d30')
            if role == TOOLTIP and column >= self.sample_start:
                return ('Selected for Plot EICs. Click to deselect.' if picked else 'Click to select this sample EIC.')
        # MatrixModel formats m/z and RT by their original column positions.
        if role == DISPLAY:
            value = self.value(row, column)
            if value is None:
                return ''
            if isinstance(value, float):
                return f'{value:.6f}' if column == 2 else f'{value:.4f}' if column == 3 else f'{value:.8g}'
            return str(value)
        return super().cell(row, column, role)

    def flags(self, index):
        flags = super().flags(index)
        return flags | Qt.ItemFlag.ItemIsUserCheckable if index.column() == 0 else flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or self.result is None or index.column() != 0 or role != Qt.ItemDataRole.CheckStateRole:
            return False
        if value == Qt.CheckState.Checked or value == Qt.CheckState.Checked.value:
            self.picks[index.row()] = set(range(len(self.result.samples)))
        else:
            self.picks.pop(index.row(), None)
        self._selection_updated(index.row())
        return True

    def toggle_cell(self, row, sample):
        selected = self.picks.setdefault(row, set())
        if sample in selected:
            selected.remove(sample)
        else:
            selected.add(sample)
        if not selected:
            self.picks.pop(row)
        self._selection_updated(row)

    def _selection_updated(self, row):
        self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1),
                              [Qt.ItemDataRole.CheckStateRole, Qt.ItemDataRole.BackgroundRole,
                               Qt.ItemDataRole.ForegroundRole, Qt.ItemDataRole.ToolTipRole])
        self.selection_changed.emit()

    def clear_picks(self):
        if not self.picks:
            return
        self.picks.clear()
        if self.rowCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))
        self.selection_changed.emit()


SORT_HINT = QAbstractItemModel.LayoutChangeHint.VerticalSortHint
NO_HINT = QAbstractItemModel.LayoutChangeHint.NoLayoutChangeHint
LAYOUT = 'QList<QPersistentModelIndex>,QAbstractItemModel::LayoutChangeHint'


def runs(positions):
    """[first, last] runs of consecutive ascending positions."""
    out = []
    for position in positions:
        if out and out[-1][1] == position-1:
            out[-1][1] = position
        else:
            out.append([position, position])
    return out


def libcxx_stable_sort(items, less):
    """std::stable_sort as compiled into Qt (libc++), comparison for comparison.

    Any stable sort gives the same order for a strict weak ordering; this port
    is only needed for keys that are not one (NaN)."""
    items = list(items)

    def merge(left, right):
        out, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if less(right[j], left[i]):
                out.append(right[j]); j += 1
            else:
                out.append(left[i]); i += 1
        return out + left[i:] + right[j:]

    def insertion_move(values):
        out = [values[0]]
        for x in values[1:]:
            n = len(out)
            if less(x, out[n-1]):
                out.append(out[n-1]); i = j = n-1
                while i:
                    i -= 1
                    if not less(x, out[i]):
                        break
                    out[j] = out[i]; j -= 1
                out[j] = x
            else:
                out.append(x)
        return out

    def sort_move(lo, hi):
        n = hi-lo
        if n <= 1:
            return items[lo:hi]
        if n == 2:
            return [items[hi-1], items[lo]] if less(items[hi-1], items[lo]) else [items[lo], items[hi-1]]
        if n <= 8:
            return insertion_move(items[lo:hi])
        middle = lo+n//2
        sort(lo, middle); sort(middle, hi)
        return merge(items[lo:middle], items[middle:hi])

    def sort(lo, hi):
        n = hi-lo
        if n == 2 and less(items[hi-1], items[lo]):
            items[lo], items[hi-1] = items[hi-1], items[lo]
        if n <= 2:
            return
        if n <= 128:
            for i in range(lo+1, hi):
                if less(items[i], items[i-1]):
                    t, j = items[i], i
                    while True:
                        items[j] = items[j-1]; j -= 1
                        if j == lo or not less(t, items[j-1]):
                            break
                    items[j] = t
            return
        middle = lo+n//2
        items[lo:hi] = merge(sort_move(lo, middle), sort_move(middle, hi))

    sort(0, len(items))
    return items


class FeatureFilter(QAbstractProxyModel):
    """Feature-row filter and sort with QSortFilterProxyModel's flat-table rules.

    A port of Qt 6's mapping: lazy creation, stable sorts of the current order,
    filter removal intervals, sorted interval insertion, dataChanged re-sorting
    and source row, column, layout and lifetime changes, with the same signals
    and persistent-index updates. Typed column keys replace two Python data()
    dispatches per comparison. Ported from and differentially tested against
    the Qt 6.11 build in PySide6 6.11.2; ``feature_filter`` uses Qt's own proxy
    on other versions. Requalify on Qt upgrades."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bounds = (None, None, None, None)
        self.text = ''
        self._rows = None  # proxy row -> source row; None until mapped
        self._columns = None  # proxy column -> source column, mapped with the rows
        self._where = None  # source row -> proxy row or -1, rebuilt on demand
        self._count = 0  # source rows when mapped
        self._sort_column = self._source_column = -1
        self._order = Qt.SortOrder.AscendingOrder
        self._connections = []
        self._saved, self._parents = [], []  # held across a source layout change

    # Structure ---------------------------------------------------------------
    def setSourceModel(self, model):
        self.beginResetModel()
        for signal, slot in self._connections:
            signal.disconnect(slot)
        super().setSourceModel(model)
        self._connections = [] if model is None else [(getattr(model, name), slot) for name, slot in (
            ('destroyed', self._source_destroyed), ('modelAboutToBeReset', self._source_resetting),
            ('modelReset', self._source_reset), ('dataChanged', self._source_data_changed),
            ('headerDataChanged', self._source_header_changed),
            ('rowsAboutToBeInserted', self._source_inserting), ('rowsInserted', self._source_rows_inserted),
            ('rowsAboutToBeRemoved', self._source_rows_removing), ('rowsRemoved', self._source_rows_removed),
            ('rowsAboutToBeMoved', self._source_rows_moving), ('rowsMoved', self._source_rows_moved),
            ('columnsAboutToBeInserted', self._source_inserting), ('columnsInserted', self._source_columns_inserted),
            ('columnsAboutToBeRemoved', self._source_columns_removing), ('columnsRemoved', self._source_columns_removed),
            ('columnsAboutToBeMoved', self._source_columns_moving), ('columnsMoved', self._source_columns_moved))]
        if model is not None:
            self._connections += [(model.layoutAboutToBeChanged[LAYOUT], self._source_layout_changing),
                                  (model.layoutChanged[LAYOUT], self._source_layout_changed)]
        for signal, slot in self._connections:
            signal.connect(slot)
        self._clear()
        self.endResetModel()
        if self._update_source_column():
            self._sort()

    def _source_destroyed(self, *_):
        # The base proxy has dropped the model and its persistent indexes; its
        # connections died with it.
        self._rows = self._columns = self._where = None
        self._connections = []

    def _source_resetting(self):
        self.beginResetModel()

    def _source_reset(self):
        self._clear()
        self.endResetModel()
        self._update_source_column()

    def _source_header_changed(self, orientation, first, last):
        self._mapping()
        if orientation == Qt.Orientation.Vertical:
            positions = sorted(position for position in self._positions()[first:last+1] if position >= 0)
        else:
            positions = [position for position, column in enumerate(self._columns) if first <= column <= last]
        for start, end in runs(positions):
            self.headerDataChanged.emit(orientation, start, end)

    def _clear(self):
        """_q_clearMapping(): drop the mapping; re-find the sort column without sorting."""
        self._rows = self._columns = self._where = None
        self._source_column = self._found_column()

    def _found_column(self):
        model = self.sourceModel()
        columns = model.columnCount() if model is not None else 0
        return self._sort_column if 0 <= self._sort_column < columns else -1

    def _mapping(self):
        if self._rows is None:
            model = self.sourceModel()
            self._count = model.rowCount() if model is not None else 0
            self._columns = list(range(model.columnCount() if model is not None else 0))
            accepts = self._acceptor()
            self._rows = self._sorted([row for row in range(self._count) if accepts(row)])
            self._where = None
        return self._rows

    def _positions(self):
        if self._where is None:
            rows = self._mapping()
            where = [-1]*self._count
            for position, row in enumerate(rows):
                where[row] = position
            self._where = where
        return self._where

    def index(self, row, column, parent=QModelIndex()):
        if row < 0 or column < 0 or parent.isValid() or row >= len(self._mapping()) or column >= len(self._columns):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, index=None):
        return QObject.parent(self) if index is None else QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._mapping())

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        self._mapping()
        return len(self._columns)

    def hasChildren(self, parent=QModelIndex()):
        return not parent.isValid() and self.rowCount() > 0 and self.columnCount() > 0

    def mapToSource(self, index):
        if not index.isValid():
            return QModelIndex()
        rows, columns = self._mapping(), self._columns
        if index.row() >= len(rows) or index.column() >= len(columns):
            return QModelIndex()
        return self.sourceModel().createIndex(rows[index.row()], columns[index.column()])

    def mapFromSource(self, index):
        if not index.isValid():
            return QModelIndex()
        where, columns = self._positions(), self._columns
        row, column = index.row(), bisect_left(columns, index.column())
        if row >= len(where) or where[row] < 0 or column == len(columns) or columns[column] != index.column():
            return QModelIndex()
        return self.createIndex(where[row], column)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        return self.sourceModel().cell(self._rows[index.row()], self._columns[index.column()], role) if index.isValid() else None

    def flags(self, index):
        model = self.sourceModel()
        return model.flags(self.mapToSource(index)) if model is not None else Qt.ItemFlag.NoItemFlags

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        rows, columns = self._mapping(), self._columns
        sections = rows if orientation == Qt.Orientation.Vertical else columns
        if 0 <= section < len(sections):
            return self.sourceModel().headerData(sections[section], orientation, role)
        # Qt defers to QAbstractProxyModel only while the proxy has cells.
        return super().headerData(section, orientation, role) if rows and columns else None

    # Source structure --------------------------------------------------------
    def _source_inserting(self, parent, *_):
        if not parent.isValid():
            self._mapping()  # as Qt: map the old rows before the insertion is announced

    def _source_rows_inserted(self, parent, start, end):
        if parent.isValid() or self._rows is None:
            return
        n = end-start+1
        self._rows[:] = [row+n if row >= start else row for row in self._rows]
        self._count += n
        self._where = None
        accepts = self._acceptor()
        self._insert(self._sorted([row for row in range(start, end+1) if accepts(row)]))
        if self._update_source_column():
            self._sort()

    def _source_rows_removing(self, parent, start, end):
        if not parent.isValid() and self._rows is not None:
            self._remove([row for row in self._rows if start <= row <= end])

    def _source_rows_removed(self, parent, start, end):
        if parent.isValid() or self._rows is None:
            return
        n = min(end, self._count-1)-start+1
        self._rows[:] = [row-n if row >= start else row for row in self._rows]
        self._count -= n
        self._where = None

    def _source_columns_inserted(self, parent, start, end):
        if parent.isValid() or self._columns is None:
            return
        n, columns = end-start+1, self._columns
        columns[:] = [column+n if column >= start else column for column in columns]
        position = bisect_left(columns, start)
        self.beginInsertColumns(QModelIndex(), position, position+n-1)
        columns[position:position] = range(start, end+1)
        self.endInsertColumns()
        if self._source_column < 0:
            if self._update_source_column():
                self._sort()
        else:
            self._source_column += n if start <= self._source_column else 0
            self._sort_column = self._mapped_sort_column()

    def _source_columns_removing(self, parent, start, end):
        if parent.isValid() or self._columns is None:
            return
        for first, last in reversed(runs([p for p, column in enumerate(self._columns) if start <= column <= end])):
            self.beginRemoveColumns(QModelIndex(), first, last)
            del self._columns[first:last+1]
            self.endRemoveColumns()

    def _source_columns_removed(self, parent, start, end):
        if parent.isValid():
            return
        n = end-start+1
        if self._columns is not None:
            self._columns[:] = [column-n if column >= start else column for column in self._columns]
        if start <= self._source_column:
            self._source_column = self._source_column-n if end < self._source_column else -1
        self._sort_column = self._mapped_sort_column()

    def _mapped_sort_column(self):
        # Qt maps the source sort column through row 0: -1 when that row is hidden.
        if self._source_column < 0:
            return -1
        return self.mapFromSource(self.sourceModel().index(0, self._source_column)).column()

    def _source_rows_moving(self, parent, start, end, destination, row):
        self._source_layout_changing(self._moved_parents(parent, destination))

    def _source_rows_moved(self, parent, start, end, destination, row):
        self._source_layout_changed(self._moved_parents(parent, destination))

    def _source_columns_moving(self, parent, start, end, destination, column):
        self._saved = []
        self.layoutAboutToBeChanged[LAYOUT].emit([QPersistentModelIndex(self.mapFromSource(p))
                                                  for p in self._moved_parents(parent, destination)], NO_HINT)
        self._saved = self._store()

    def _source_columns_moved(self, parent, start, end, destination, column):
        parents = [QPersistentModelIndex(self.mapFromSource(p)) for p in self._moved_parents(parent, destination)]
        self._rows = self._columns = self._where = None
        self._restore()
        self._update_source_column()
        self.layoutChanged[LAYOUT].emit(parents, NO_HINT)

    @staticmethod
    def _moved_parents(parent, destination):
        return [QPersistentModelIndex(parent)] + ([QPersistentModelIndex(destination)] if destination != parent else [])

    def _source_layout_changing(self, parents, hint=None):
        # Like Qt, drop the hint: filtering and sorting may move other rows too.
        self._saved = []
        mapped = ((parent, QPersistentModelIndex(self.mapFromSource(parent))) for parent in parents)
        self._parents = [proxy for parent, proxy in mapped if proxy.isValid() or not parent.isValid()]
        if parents and not self._parents:
            return
        self.layoutAboutToBeChanged[LAYOUT].emit(self._parents, NO_HINT)
        self._saved = self._store()

    def _source_layout_changed(self, parents, hint=None):
        if parents and not self._parents:
            return
        self._rows = self._columns = self._where = None
        self._restore()
        self._source_column = self._found_column()
        self.layoutChanged[LAYOUT].emit(self._parents, NO_HINT)

    def _store(self):
        """store_persistent_indexes(): each proxy index with the source cell it shows."""
        return [(index, QPersistentModelIndex(self.mapToSource(index))) for index in self.persistentIndexList()]

    def _restore(self):
        """update_persistent_indexes() through a fresh mapping (made with the old sort column)."""
        saved, self._saved = self._saved, []
        if saved:
            self._mapping()
            self.changePersistentIndexList([index for index, _ in saved], [self.mapFromSource(source) for _, source in saved])

    # Filtering ---------------------------------------------------------------
    def set_filters(self, text, bounds):
        self._mapping()  # as QSortFilterProxyModel.beginFilterChange()
        self.text, self.bounds = text.strip().lower(), bounds
        accepts = self._acceptor()
        remove = [row for row in self._rows if not accepts(row)]
        where = self._positions()
        insert = [row for row in range(self._count) if where[row] < 0 and accepts(row)]
        if remove or insert:
            self._remove(remove)
            self._insert(self._sorted(insert))

    def filterAcceptsRow(self, row, parent):
        return self._acceptor()(row)

    def _acceptor(self):
        """filterAcceptsRow() for many rows, with the model and filter looked up once."""
        model = self.sourceModel()
        if model is None or model.result is None:
            return lambda row: False
        features, (lo, hi, start, end), text = model.result.features, self.bounds, self.text
        def accepts(row):
            f = features[row]
            if ((lo is not None and f.mz < lo) or (hi is not None and f.mz > hi)
                    or (start is not None and f.rt < start) or (end is not None and f.rt > end)):
                return False
            return not text or text in f'{f.id} {f.mz:.6f} {f.rt:.4f}'.lower()
        return accepts

    # Sorting -----------------------------------------------------------------
    def sortColumn(self):
        return self._sort_column

    def sortOrder(self):
        return self._order

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        order = Qt.SortOrder(order)
        if column == self._sort_column and order == self._order:
            return
        self._order, self._sort_column = order, column
        self._update_source_column()
        self._sort()

    def _update_source_column(self):
        old = self._source_column
        if self._sort_column == -1:
            self._source_column = -1
        else:
            self._mapping()  # created with the previous sort column, as in Qt
            columns = self._columns
            self._source_column = columns[self._sort_column] if 0 <= self._sort_column < len(columns) else -1
        return old != self._source_column

    def _sort(self):
        def change():
            if self._rows is not None:
                self._rows[:] = self._sorted(self._rows)
        self._relayout(change)

    def _relayout(self, change, parents=()):
        self.layoutAboutToBeChanged[LAYOUT].emit(list(parents), SORT_HINT)
        old = self.persistentIndexList()
        sources = [(self._rows[index.row()], index.column()) for index in old]
        change()
        self._where = None
        where = self._positions()
        self.changePersistentIndexList(old, [self.createIndex(where[row], column) if where[row] >= 0 else QModelIndex()
                                             for row, column in sources])
        self.layoutChanged[LAYOUT].emit(list(parents), SORT_HINT)

    def _less(self, keys):
        ascending = self._order == Qt.SortOrder.AscendingOrder
        def less(left, right):
            a, b = keys[left], keys[right]
            if a is None or b is None:
                if a is b:
                    return left < right
                return (a is not None) if ascending else (a is None)
            return a < b
        return less

    def _keys(self):
        model = self.sourceModel()
        # A column kept across a reset to fewer columns reads invalid indexes: all ties.
        return model.column_values(self._source_column) if self._source_column < model.columnCount() else None

    def _comparator(self):
        """Source-row order used by std::stable_sort and interval insertion."""
        if self._source_column < 0:
            return int.__gt__ if self._order == Qt.SortOrder.DescendingOrder else int.__lt__
        keys = self._keys()
        if keys is None:
            return lambda a, b: False
        less = self._less(keys)
        return (lambda a, b: less(b, a)) if self._order == Qt.SortOrder.DescendingOrder else less

    def _sorted(self, rows):
        """sort_source_rows(): stable sort of source rows in their current order."""
        descending = self._order == Qt.SortOrder.DescendingOrder
        if not rows:
            return []
        if self._source_column < 0:
            return sorted(rows, reverse=descending)
        keys = self._keys()
        if keys is None:
            return list(rows)
        present = [row for row in rows if keys[row] is not None]
        if any(keys[row] != keys[row] for row in present):
            return libcxx_stable_sort(rows, self._comparator())
        present.sort(key=keys.__getitem__, reverse=descending)
        return present + sorted((row for row in rows if keys[row] is None), reverse=descending)

    # Row intervals -----------------------------------------------------------
    def _remove(self, source_rows, signals=True):
        where = self._positions()
        for start, end in reversed(runs(sorted(where[row] for row in source_rows))):
            if signals:
                self.beginRemoveRows(QModelIndex(), start, end)
            del self._rows[start:end+1]
            self._where = None
            if signals:
                self.endRemoveRows()

    def _insert(self, source_rows, signals=True):
        """Insert sorted rows by proxy_intervals_for_source_items_to_add()."""
        rows = self._rows
        before = self._comparator() if self._source_column >= 0 else int.__lt__
        intervals, low, i = [], 0, 0
        while i < len(source_rows):
            first = source_rows[i]; interval = [first]; i += 1
            high = len(rows)-1
            while low <= high:
                middle = (low+high)//2
                if before(first, rows[middle]):
                    high = middle-1
                else:
                    low = middle+1
            if low >= len(rows):
                interval += source_rows[i:]; i = len(source_rows)
            else:
                anchor = rows[low]
                while i < len(source_rows) and not before(anchor, source_rows[i]):
                    interval.append(source_rows[i]); i += 1
            intervals.append((low, interval))
        for start, interval in reversed(intervals):
            if signals:
                self.beginInsertRows(QModelIndex(), start, start+len(interval)-1)
            rows[start:start] = interval
            self._where = None
            if signals:
                self.endInsertRows()

    def _source_data_changed(self, top_left, bottom_right, roles=()):
        if not top_left.isValid() or not bottom_right.isValid():
            return
        unmapped = self._rows is None  # Qt maps it now and forwards the change as is
        accepts, where = self._acceptor(), self._positions()
        # Qt 6 re-sorts on any change spanning the sort column, whatever the roles.
        resorting = not unmapped and top_left.column() <= self._source_column <= bottom_right.column()
        remove, insert, change, resort = [], [], [], []
        for row in range(top_left.row(), bottom_right.row()+1):
            if where[row] >= 0:
                if not unmapped and not accepts(row):
                    remove.append(row)
                else:
                    (resort if resorting else change).append(row)
            elif not unmapped and accepts(row):
                insert.append(row)
        if remove:
            self._remove(remove)
        if resort:
            if self._needs_reorder(resort):
                def reorder():
                    self._remove(resort, False)
                    self._insert(self._sorted(resort), False)
                self._relayout(reorder, [QPersistentModelIndex()])
            change += resort
        if change:
            where = self._positions()
            positions = [where[row] for row in change]
            self.dataChanged.emit(self.index(min(positions), top_left.column()),
                                  self.index(max(positions), bottom_right.column()), roles)
        if insert:
            self._insert(self._sorted(insert))

    def _needs_reorder(self, source_rows):
        rows, where = self._rows, self._positions()
        if not 0 <= self._sort_column < len(self._columns):
            # Qt compares with neighbours in the proxy sort column. After a column
            # change hid row 0 that is -1: invalid indexes, no value, row -1.
            keys, ascending = self._keys(), self._order == Qt.SortOrder.AscendingOrder
            return any(where[row] > 0 and (not ascending or keys[row] is not None)
                       or ascending and where[row] < len(rows)-1 and keys[row] is None for row in source_rows)
        before = self._comparator()
        for row in source_rows:
            position = where[row]
            if position > 0 and before(row, rows[position-1]):
                return True
            if position < len(rows)-1 and before(rows[position+1], row):
                return True
        return False


class QtFeatureFilter(QSortFilterProxyModel):
    """The original Heights proxy: Qt's own mapping with per-comparison data() calls."""
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


def feature_filter(parent=None):
    """The Heights proxy: the port on the Qt it reproduces, else Qt's own (slower) proxy."""
    return (FeatureFilter if qualified("Qt", "PySide6") else QtFeatureFilter)(parent)


class Table(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setShowGrid(False)
        self.verticalHeader().hide()
        self.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            indexes = self.selectedIndexes()
            if indexes:
                selected = {(i.row(), i.column()): i.data() for i in indexes}
                rows, columns = sorted({i.row() for i in indexes}), sorted({i.column() for i in indexes})
                text = StringIO()
                writer = csv.writer(text, delimiter='\t', lineterminator='\n')
                writer.writerows([[selected.get((r,c), '') for c in columns] for r in rows])
                QApplication.clipboard().setText(text.getvalue().rstrip('\n'))
            return
        super().keyPressEvent(event)


class FeatureTable(Table):
    """Keep the four identifying columns visible while samples scroll."""
    def __init__(self, *, frozen_columns=4):
        super().__init__()
        self.frozen_columns = frozen_columns
        self.frozen = Table(self)
        # Opaque, so the main viewport skips the identifying cells painted beneath it.
        self.frozen.setAutoFillBackground(True)
        self.frozen.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frozen.setSortingEnabled(False)
        self.frozen.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().stackUnder(self.frozen)
        self.verticalScrollBar().valueChanged.connect(self.frozen.verticalScrollBar().setValue)
        self.frozen.verticalScrollBar().valueChanged.connect(self.verticalScrollBar().setValue)
        self.horizontalHeader().sectionResized.connect(self._width_changed)
        self.frozen.horizontalHeader().sectionResized.connect(lambda c,old,new:self.setColumnWidth(c,new) if c<self.frozen_columns else None)
        self.frozen.horizontalHeader().setSortIndicatorShown(True)
        self.horizontalHeader().sortIndicatorChanged.connect(lambda column,order:self.frozen.horizontalHeader().setSortIndicator(column,order))
        self.frozen.horizontalHeader().sectionClicked.connect(self._sort_frozen)

    def _sort_frozen(self, column):
        header=self.horizontalHeader()
        order=Qt.SortOrder.DescendingOrder if header.sortIndicatorSection()==column and header.sortIndicatorOrder()==Qt.SortOrder.AscendingOrder else Qt.SortOrder.AscendingOrder
        self.sortByColumn(column,order)

    def setModel(self, model):
        super().setModel(model)
        self.frozen.setModel(model);self.frozen.setSelectionModel(self.selectionModel())
        model.modelReset.connect(self._refresh_frozen)
        self._refresh_frozen()

    def _refresh_frozen(self):
        for column in range(self.model().columnCount()):
            self.frozen.setColumnHidden(column,column>=self.frozen_columns)
            if column<self.frozen_columns:self.frozen.setColumnWidth(column,self.columnWidth(column))
        self._geometry()

    def _width_changed(self,column,old,new):
        if column<self.frozen_columns:self.frozen.setColumnWidth(column,new);self._geometry()

    def _geometry(self):
        width=sum(self.columnWidth(c) for c in range(min(self.frozen_columns,self.model().columnCount())))
        self.frozen.horizontalHeader().setFixedHeight(self.horizontalHeader().height())
        self.frozen.setGeometry(self.frameWidth(),self.frameWidth(),width,
                                self.viewport().height()+self.horizontalHeader().height())

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'frozen') and self.model() is not None:self._geometry()


class EICFeatureTable(FeatureTable):
    def __init__(self):
        super().__init__(frozen_columns=5)

    def _sort_frozen(self, column):
        if column:
            super()._sort_frozen(column)

    def mousePressEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and index.isValid() and index.column() >= 5:
            self.clearSelection()
            self.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            source = self.model().mapToSource(index)
            self.model().sourceModel().toggle_cell(source.row(), source.column()-5)
            self.setFocus()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # A second press in a double click is another toggle, never an edit.
        self.mousePressEvent(event)

    def keyPressEvent(self, event):
        index = self.currentIndex()
        if event.key() == Qt.Key.Key_Space and index.isValid() and index.column() >= 5:
            source = self.model().mapToSource(index)
            self.model().sourceModel().toggle_cell(source.row(), source.column()-5)
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Copy) and not self.selectedIndexes():
            model = self.model().sourceModel()
            cells = model.selected_cells
            if cells:
                selected = set(cells)
                rows = sorted({r for r, s in cells})
                samples = sorted({s for r, s in cells})
                text = StringIO()
                csv.writer(text, delimiter='\t', lineterminator='\n').writerows(
                    [[model.index(r, s+5).data() if (r, s) in selected else '' for s in samples] for r in rows])
                QApplication.clipboard().setText(text.getvalue().rstrip('\n'))
                return
        super().keyPressEvent(event)


class RowsModel(QAbstractTableModel):
    def __init__(self, headers, rows):
        super().__init__()
        self.headers, self.rows = headers, rows

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == ROLE:
            return value
        if role == Qt.ItemDataRole.DisplayRole:
            return '' if value is None else f'{value:.8g}' if isinstance(value,float) else str(value)


class MeasurementsModel(RowsModel):
    def __init__(self, result):
        from dataclasses import fields
        from kmzcomet.model import Measurement
        self.result = result
        self.names = [f.name for f in fields(Measurement)]
        # Rows are generated on access; large analyses do not duplicate cells.
        super().__init__([name.replace('_',' ').capitalize() for name in self.names], ())

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.result.features)*len(self.result.samples)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        sample, feature = divmod(index.row(), len(self.result.features))
        value = getattr(self.result.measurements[sample][feature], self.names[index.column()])
        if role == ROLE:
            return value
        if role == Qt.ItemDataRole.DisplayRole:
            return '' if value is None else f'{value:.8g}' if isinstance(value,float) else str(value)


def run_rows(result):
    rows = [('Analysis', 'ID', result.analysis_id), ('Analysis', 'Created UTC', result.created_utc),
            ('Analysis', 'Method', result.method), ('Analysis', 'Suite version', result.suite_version),
            ('Analysis', 'Elapsed seconds', result.elapsed_seconds)]
    rows += [('Settings', key, value) for key,value in asdict(result.settings).items()]
    for sample in result.samples:
        rows += [(sample.id, 'Filename', sample.path.name), (sample.id, 'Stored sample ID', sample.native_sample_id),
                 (sample.id, 'SHA-256', sample.sha256), (sample.id, 'MS1 scans', sample.scans),
                 (sample.id, 'RT shifts (min)', str(sample.shifts)), (sample.id, 'RT anchors', str(sample.anchors)),
                 (sample.id, 'RT anchor MAD (min)', str(sample.alignment_mad)),
                 (sample.id, 'Warnings', '; '.join(sample.warnings) or 'None')]
        rows += [(sample.id, 'Source file', f"{s['name']} · {s['sha256']}") for s in sample.source_files]
    from kmzcomet.excel import DEFINITIONS
    rows += [('Definition', name, text) for name,text in DEFINITIONS.items()]
    return rows
