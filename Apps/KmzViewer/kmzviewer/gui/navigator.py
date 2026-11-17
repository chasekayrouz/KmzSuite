"""Multi-selection navigator with file-only, stable-ID drag ordering."""
from PySide6.QtCore import QMimeData, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QPainter, QPen
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget


class DataNavigator(QTreeWidget):
    files_moved = Signal(list, int)
    MIME = "application/x-kmzviewer-files"

    def __init__(self):
        super().__init__()
        self._token = str(id(self))
        self.grouping = "file"
        self._drag_file = False
        self._drop_before = None
        self._drag_y = 0
        self._scroll_direction = 0
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(60)
        self._scroll_timer.timeout.connect(self._scroll_drag)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(False)
        self.setToolTip("Shift-click to select a range; Command/Ctrl-click to select individual rows. "
                        "Right-click for group actions. Drag file names to reorder.")

    def file_ids(self):
        return [self.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)[1]
                for i in range(self.topLevelItemCount())
                if self.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)[0] == "file"]

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        self._drag_file = (self.grouping == "file" and item is not None and item.parent() is None
                           and self.columnAt(int(event.position().x())) == 0)
        super().mousePressEvent(event)

    def mime_for_files(self, file_ids):
        mime = QMimeData()
        mime.setData(self.MIME, (self._token + ":" + ",".join(map(str, file_ids))).encode())
        return mime

    def startDrag(self, actions):
        if not self._drag_file:
            return
        ids = [self.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)[1]
               for i in range(self.topLevelItemCount()) if self.topLevelItem(i).isSelected()]
        if not ids:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mime_for_files(ids))
        drag.exec(Qt.DropAction.MoveAction)
        self._clear_drop()

    def _source_files(self, event):
        if self.grouping != "file":
            return []
        payload = bytes(event.mimeData().data(self.MIME)).decode(errors="replace")
        if not payload.startswith(self._token + ":"):
            return []
        try:
            ids = [int(value) for value in payload.split(":", 1)[1].split(",")]
        except ValueError:
            return []
        return ids if ids and set(ids).issubset(self.file_ids()) else []

    def _insertion_at(self, y):
        item = self.itemAt(0, y)
        if item is None:
            return self.topLevelItemCount() if y >= 0 else 0
        top = item.parent() or item
        index = self.indexOfTopLevelItem(top)
        # Children are never reparented: a drop over them follows their whole file.
        return index + (item is not top or y > self.visualItemRect(top).center().y())

    def dragEnterEvent(self, event):
        if self._source_files(event):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not self._source_files(event):
            self._clear_drop()
            event.ignore()
            return
        self._drag_y = int(event.position().y())
        self._drop_before = self._insertion_at(self._drag_y)
        self._scroll_direction = -1 if self._drag_y < 28 else (1 if self._drag_y > self.viewport().height() - 28 else 0)
        if self._scroll_direction:
            if not self._scroll_timer.isActive():
                self._scroll_timer.start()
        else:
            self._scroll_timer.stop()
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()
        self.viewport().update()

    def _scroll_drag(self):
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() + self._scroll_direction)
        self._drop_before = self._insertion_at(self._drag_y)
        self.viewport().update()

    def _clear_drop(self):
        self._scroll_timer.stop()
        self._drop_before = None
        self.viewport().update()

    def dragLeaveEvent(self, event):
        self._clear_drop()
        event.accept()

    def dropEvent(self, event):
        ids = self._source_files(event)
        if ids and self._drop_before is not None:
            self.files_moved.emit(ids, self._drop_before)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()
        self._clear_drop()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_before is None or not self.topLevelItemCount():
            return
        if self._drop_before < self.topLevelItemCount():
            y = self.visualItemRect(self.topLevelItem(self._drop_before)).top()
        else:
            last = self.topLevelItem(self.topLevelItemCount() - 1)
            if last.isExpanded() and last.childCount():
                last = last.child(last.childCount() - 1)
            y = self.visualItemRect(last).bottom()
        painter = QPainter(self.viewport())
        painter.setPen(QPen(QColor("#216653"), 2))
        painter.drawLine(0, y, self.viewport().width(), y)
        painter.end()
