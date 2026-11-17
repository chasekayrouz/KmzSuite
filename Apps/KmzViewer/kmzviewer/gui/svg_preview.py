"""Live SVG preview. The displayed bytes are also the saved artifact."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle

from PySide6.QtCore import QByteArray, QMimeData, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QPainter, QPen
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QColorDialog, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from kmzviewer.export_svg import ExportOptions, legend_font_properties, render_with_layout, suggested_y_interval
from kmzcore.trace_style import FILL_ALPHA


class NumberEdit(QLineEdit):
    """Counts fields accept scientific notation without fixed-decimal rounding."""
    def __init__(self, value, label):
        super().__init__()
        self.label = label
        self.setAccessibleName(label)
        self.setValue(value)
        self.setMinimumWidth(90)

    def value(self):
        try:
            return float(self.text().strip())
        except ValueError:
            raise ValueError(f"Enter a number for {self.label.lower()} (for example, 1e5).") from None

    def setValue(self, value):
        self.setText(f"{value:.9g}")

    def wheelEvent(self, event):
        event.ignore()


class PreviewSpinBox(QDoubleSpinBox):
    def sizeHint(self):
        return QSize(140, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(90, super().minimumSizeHint().height())

    def wheelEvent(self, event):
        # Scrolling the settings pane should not silently change plot settings.
        event.ignore()


class TraceTable(QTableWidget):
    orderChanged = Signal()
    MIME = "application/x-kmzviewer-svg-trace"

    def __init__(self, rows, data):
        super().__init__(len(rows), 2)
        self._drag_column = -1
        self._drop_before = None
        self._token = str(id(self))
        self._scroll_direction = 0
        self._drag_y = 0
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(80)
        self._scroll_timer.timeout.connect(self._scroll_drag)
        self.setHorizontalHeaderLabels(["Show / chromatogram", "Color"])
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(1, 110)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(False)
        self.setAutoScroll(True)
        self.setAutoScrollMargin(28)
        self.setMinimumHeight(140)
        self.setToolTip("Drag a name up or down to reorder the stack. Alt+↑ / Alt+↓ also move the selected row. Click a color to change it.")
        for row, (trace_id, label, checked) in enumerate(rows):
            name = QTableWidgetItem(label)
            name.setData(Qt.ItemDataRole.UserRole, trace_id)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable |
                          Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
            name.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            name.setToolTip(label)
            color = data[trace_id].color if trace_id in data else "#0072b2"
            swatch = QTableWidgetItem(color)
            swatch.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            swatch.setData(Qt.ItemDataRole.DecorationRole, QColor(color))
            swatch.setToolTip("Click to change this chromatogram’s export color")
            self.setItem(row, 0, name)
            self.setItem(row, 1, swatch)

    def mousePressEvent(self, event):
        self._drag_column = self.columnAt(int(event.position().x()))
        super().mousePressEvent(event)

    def mime_for_row(self, row):
        mime = QMimeData()
        trace_id = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
        mime.setData(self.MIME, f"{self._token}:{trace_id}".encode())
        return mime

    def startDrag(self, actions):
        if self._drag_column != 0 or self.currentRow() < 0:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mime_for_row(self.currentRow()))
        drag.setPixmap(self.viewport().grab(self.visualItemRect(self.item(self.currentRow(), 0))))
        drag.exec(Qt.DropAction.MoveAction)
        self._scroll_timer.stop()
        self._drop_before = None
        self.viewport().update()

    def _source_row(self, event):
        payload = bytes(event.mimeData().data(self.MIME)).decode(errors="replace")
        if not payload.startswith(self._token + ":"):
            return None
        identity = payload.split(":", 1)[1]
        for row in range(self.rowCount()):
            if str(self.item(row, 0).data(Qt.ItemDataRole.UserRole)) == identity:
                return row
        return None

    def dragEnterEvent(self, event):
        if self._source_row(event) is not None:
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._source_row(event) is None or self.columnAt(int(event.position().x())) != 0:
            self._drop_before = None
            self._scroll_timer.stop()
            event.ignore()
        else:
            y = int(event.position().y())
            row = self.rowAt(y)
            self._drop_before = self.rowCount() if row < 0 else row + (y > self.visualItemRect(self.item(row, 0)).center().y())
            self._drag_y = y
            self._scroll_direction = -1 if y < 28 else (1 if y > self.viewport().height() - 28 else 0)
            if self._scroll_direction:
                if not self._scroll_timer.isActive():
                    self._scroll_timer.start()
            else:
                self._scroll_timer.stop()
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        self.viewport().update()

    def _scroll_drag(self):
        scroll = self.verticalScrollBar()
        scroll.setValue(scroll.value() + self._scroll_direction)
        row = self.rowAt(self._drag_y)
        self._drop_before = self.rowCount() if row < 0 else row + (
            self._drag_y > self.visualItemRect(self.item(row, 0)).center().y())
        self.viewport().update()

    def dragLeaveEvent(self, event):
        self._scroll_timer.stop()
        self._drop_before = None
        self.viewport().update()
        event.accept()

    def dropEvent(self, event):
        self._scroll_timer.stop()
        source = self._source_row(event)
        if source is not None and self._drop_before is not None:
            self.move_row(source, self._drop_before)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()
        self._drop_before = None
        self.viewport().update()

    def move_row(self, source, before):
        if not 0 <= source < self.rowCount() or not 0 <= before <= self.rowCount():
            return
        destination = before - (before > source)
        if source == destination:
            return
        with QSignalBlocker(self):
            items = [self.takeItem(source, col) for col in range(self.columnCount())]
            self.removeRow(source)
            self.insertRow(destination)
            for col, item in enumerate(items):
                self.setItem(destination, col, item)
            self.setCurrentCell(destination, 0)
        self.orderChanged.emit()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.AltModifier and self.currentRow() >= 0:
            row = self.currentRow()
            if event.key() == Qt.Key.Key_Up and row > 0:
                self.move_row(row, row - 1)
                return
            if event.key() == Qt.Key.Key_Down and row < self.rowCount() - 1:
                self.move_row(row, row + 2)
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_before is not None and self.rowCount():
            row = min(self._drop_before, self.rowCount() - 1)
            rect = self.visualItemRect(self.item(row, 0))
            y = rect.bottom() if self._drop_before == self.rowCount() else rect.top()
            painter = QPainter(self.viewport())
            painter.setPen(QPen(QColor("#216653"), 2))
            painter.drawLine(rect.left(), y, rect.right(), y)
            painter.end()


class ExportDialog(QDialog):
    def __init__(self, traces, current_x, full_x=None, *, trace_data=None,
                 single_trace=False, default_path="chromatograms.svg", parent=None, spectrum=None, initial_layout='stacked'):
        super().__init__(parent)
        self.setWindowTitle("SVG export preview")
        self.resize(1200, 800)
        self._data = trace_data or {}
        self.spectrum_kind = spectrum
        self._primary_unit = 'counts' if any(t.unit == 'counts' for t in self._data.values()) else 'mAU'
        self._y_interval_edited = False
        self._legend_font_edited = False
        self.svg_bytes = b""
        self._rendered = None
        self.saved_path = None
        self.default_path = str(default_path)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.update_preview)
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(['Stacked', 'Overlay'])
        self.layout_choice.setCurrentText(initial_layout.title())
        self.layout_choice.currentTextChanged.connect(self.schedule_preview)

        self.width = self._spin(7.2, .5, 40, 2, .1)
        self.height = self._spin(4.8, .5, 40, 2, .1)
        self.font_size = self._spin(12, 4, 72, 1, .5)
        self.tick_font_size = self._spin(10, 4, 72, 1, .5)
        self.legend_font_size = self._spin(10, .1, 72, 2, .5)
        self.legend_font_size.setToolTip("Starts at 10 pt or smaller to fit full names to the right of the plot. Follows canvas and selection changes until you edit it; your size then stays fixed.")
        self.xmin, self.xmax = self._spin(current_x[0]), self._spin(current_x[1])
        self.line_width = self._spin(2, .1, 20, 2, .1)
        self.shading = QCheckBox("Under-curve shading")
        self.shading.setChecked(True)
        self.opacity = self._spin(round(255 * FILL_ALPHA) / 255 * 100, 0, 100, 6, 1)
        self.auto_spacing = QCheckBox("Auto spacing: visible signal span × 1.1")
        self.auto_spacing.setChecked(True)
        self.spacing = NumberEdit(0, "Stack spacing")
        self.spacing.setEnabled(False)
        self.offset = self._spin(0)
        self.offset.setToolTip("The bottom checked trace stays fixed. Each trace above it shifts by one more offset; positive values stack up and right.")
        self.legend = QCheckBox("Show legend")
        self.legend.setChecked(True)
        self.auto_y = QCheckBox("Automatic Y range")
        self.auto_y.setChecked(True)
        self.ymin, self.ymax = NumberEdit(0, "Y minimum"), NumberEdit(1, "Y maximum")
        self.ymin.setEnabled(False)
        self.ymax.setEnabled(False)
        self.x_interval = self._spin(50 if spectrum else 1, .000001, 1e9)
        self.x_interval.setToolTip("RT ticks repeat from zero minutes at this interval.")
        self.y_interval = NumberEdit(1, "Y tick interval")
        self.y_interval.setToolTip("Starts with a sparse interval suited to the Y range. Enter an interval to keep it fixed; ticks start at zero. Scientific notation such as 1e5 is accepted.")
        self.grid = QCheckBox("Faint vertical gridlines")
        self.grid_interval = self._spin(1, .000001, 1e9)
        self.grid_interval.setEnabled(False)
        self.grid_interval.setToolTip("Spacing in minutes at the bottom baseline. Gridlines follow the angle of offset stacks.")

        self.right_auto_y = QCheckBox('Automatic absorbance range')
        self.right_auto_y.setChecked(True)
        self.right_ymin, self.right_ymax = NumberEdit(0, 'Absorbance minimum'), NumberEdit(1, 'Absorbance maximum')
        self.right_ymin.setEnabled(False)
        self.right_ymax.setEnabled(False)
        self.right_auto_spacing = QCheckBox('Automatic absorbance spacing')
        self.right_auto_spacing.setChecked(True)
        self.right_spacing = NumberEdit(0, 'Absorbance spacing')
        self.right_spacing.setEnabled(False)
        self.right_y_interval = NumberEdit(1, 'Absorbance tick interval')
        self._right_interval_edited = False
        self.right_y_interval.textEdited.connect(lambda: setattr(self, '_right_interval_edited', True))
        self.right_auto_y.toggled.connect(lambda checked: (self.right_ymin.setEnabled(not checked), self.right_ymax.setEnabled(not checked)))
        self.right_auto_spacing.toggled.connect(lambda checked: self.right_spacing.setEnabled(not checked))
        form = self.form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for label, control in (
            ('Layout', self.layout_choice),
            ("Canvas width (in)", self.width), ("Canvas height (in)", self.height),
            ("Axis labels (pt)", self.font_size), ("Tick text (pt)", self.tick_font_size),
            ("Legend font size (pt)", self.legend_font_size),
            ("RT minimum (min)", self.xmin), ("RT maximum (min)", self.xmax),
            ("Line weight (pt)", self.line_width), ("", self.shading),
            ("Fill opacity (%)", self.opacity), ("", self.auto_spacing),
            ("Stack spacing (counts)", self.spacing), ("Successive RT offset (min)", self.offset),
            ("", self.auto_y), ("Y minimum", self.ymin), ("Y maximum", self.ymax),
            ("RT tick interval (min)", self.x_interval), ("Y tick interval (counts)", self.y_interval),
            ("", self.grid), ("Grid spacing (min)", self.grid_interval),
            ('', self.right_auto_y), ('Absorbance minimum (mAU)', self.right_ymin), ('Absorbance maximum (mAU)', self.right_ymax),
            ('', self.right_auto_spacing), ('Absorbance spacing (mAU)', self.right_spacing), ('Absorbance tick interval (mAU)', self.right_y_interval),
        ):
            if label:
                form.addRow(label, control)
                control.setAccessibleName(label)
            else:
                form.addRow(control)

        self.table = TraceTable(traces, self._data)
        if spectrum:
            self.table.setHorizontalHeaderLabels(['Show / spectrum', 'Color'])
            self.shading.setChecked(False)
        self.table.cellClicked.connect(self._choose_color)
        self.table.itemChanged.connect(self.schedule_preview)
        self.table.orderChanged.connect(self.schedule_preview)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.addLayout(form)
        settings_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        scroll.setMinimumWidth(350)
        self.preview = QSvgWidget()
        self.preview.setMinimumSize(240, 180)
        self.preview.setStyleSheet("background: white;")
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        caption = QLabel("Transparent SVG · white preview surface · native units")
        caption.setWordWrap(True)
        preview_header = QHBoxLayout()
        preview_header.addWidget(caption, 1)
        preview_header.addWidget(self.legend)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.preview, 1)
        hint = QLabel("Drag names to reorder · table order is top to bottom · mixed traces use separate counts and absorbance axes")
        hint.setWordWrap(True)
        preview_layout.addWidget(hint)
        preview_layout.addWidget(self.table)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(preview_panel)
        splitter.setSizes([390, 810])
        splitter.setChildrenCollapsible(False)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #9b2c2c")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.save_button = QPushButton("Save SVG…")
        self.save_button.setAutoDefault(False)
        self.save_button.clicked.connect(self.save_svg)
        close = QPushButton("Close")
        close.setAutoDefault(False)
        close.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.summary, 1)
        buttons.addWidget(close)
        buttons.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.error)
        layout.addLayout(buttons)
        self.auto_spacing.toggled.connect(lambda checked: self.spacing.setEnabled(not checked))
        self.auto_y.toggled.connect(lambda checked: (self.ymin.setEnabled(not checked), self.ymax.setEnabled(not checked)))
        self.shading.toggled.connect(self.opacity.setEnabled)
        self.grid.toggled.connect(self.grid_interval.setEnabled)
        self.y_interval.textEdited.connect(self._custom_y_interval)
        self.legend_font_size.valueChanged.connect(self._custom_legend_font)
        for control in self.findChildren(QDoubleSpinBox):
            control.valueChanged.connect(self.schedule_preview)
        for control in self.findChildren(NumberEdit):
            control.textChanged.connect(self.schedule_preview)
        for control in (self.auto_spacing, self.auto_y, self.shading, self.legend, self.grid, self.right_auto_y, self.right_auto_spacing):
            control.toggled.connect(self.schedule_preview)
        self.update_preview()

    @staticmethod
    def _spin(value, low=-1e15, high=1e15, decimals=6, step=1):
        spin = PreviewSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _custom_y_interval(self, *_):
        self._y_interval_edited = True

    def _custom_legend_font(self, *_):
        self._legend_font_edited = True

    def _choose_color(self, row, column):
        if column != 1:
            return
        item = self.table.item(row, column)
        color = QColorDialog.getColor(QColor(item.text()), self, "Chromatogram color")
        if color.isValid():
            with QSignalBlocker(self.table):
                item.setText(color.name())
                item.setData(Qt.ItemDataRole.DecorationRole, color)
            self.schedule_preview()

    def selected_trace_ids(self):
        return [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                for row in range(self.table.rowCount())
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]

    def _selection(self):
        rows = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            identity = item.data(Qt.ItemDataRole.UserRole)
            if item.checkState() == Qt.CheckState.Checked and identity in self._data:
                rows.append((identity, self.table.item(row, 1).text()))
        return rows

    def selected_traces(self):
        return [replace(self._data[identity], color=color) for identity, color in self._selection()]

    def export_options(self):
        overlay = self.layout_choice.currentText() == 'Overlay'
        return ExportOptions(
            layout=self.layout_choice.currentText().lower(),
            x_label='Wavelength (nm)' if self.spectrum_kind == 'dad' else 'm/z' if self.spectrum_kind else 'Retention time (min)',
            x_unit='nm' if self.spectrum_kind == 'dad' else 'm/z' if self.spectrum_kind else 'min',
            spectrum=bool(self.spectrum_kind),
            right_auto_y=self.right_auto_y.isChecked(), right_ymin=self.right_ymin.value(), right_ymax=self.right_ymax.value(),
            right_spacing=None if overlay or self.right_auto_spacing.isChecked() else self.right_spacing.value(),
            right_y_tick_interval=self.right_y_interval.value() if self._right_interval_edited else None,
            width_in=self.width.value(), height_in=self.height.value(), fontsize=self.font_size.value(),
            tick_fontsize=self.tick_font_size.value(), xmin=self.xmin.value(), xmax=self.xmax.value(),
            linewidth=self.line_width.value(), fill=self.shading.isChecked(), opacity=self.opacity.value()/100,
            auto_y=self.auto_y.isChecked(), ymin=None if self.auto_y.isChecked() else self.ymin.value(),
            ymax=None if self.auto_y.isChecked() else self.ymax.value(), legend=self.legend.isChecked(),
            spacing=None if overlay or self.auto_spacing.isChecked() else self.spacing.value(), offset=0 if overlay else self.offset.value(),
            x_tick_interval=self.x_interval.value(),
            y_tick_interval=self.y_interval.value() if self._y_interval_edited else None,
            grid=self.grid.isChecked(), grid_interval=self.grid_interval.value(),
            legend_fontsize=self.legend_font_size.value() if self._legend_font_edited else None,
        )

    def schedule_preview(self, *_):
        self.save_button.setEnabled(False)
        self._timer.start()

    def update_preview(self):
        self._timer.stop()
        try:
            selection = self._selection()
            traces = [replace(self._data[identity], color=color) for identity, color in selection]
            self._update_axis_controls(traces)
            options = self.export_options()
            # Exact inputs (pickled floats keep signed zeros) and the legend's
            # font context: an unchanged state, such as Save after its preview,
            # reuses the same bytes.
            key = pickle.dumps((selection, options)), options.legend and legend_font_properties()[1]
            if self._rendered is None or self._rendered[0] != key:
                rendered = render_with_layout(traces, options)
                if rendered[0] != self.svg_bytes:
                    self.preview.load(QByteArray(rendered[0]))
                    self.preview.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
                    if not self.preview.renderer().isValid():
                        raise ValueError("The SVG could not be rendered.")
                self._rendered = key, rendered
            svg, legend_font, spacings, axes = self._rendered[1]
            self.svg_bytes = svg
            if options.legend and options.legend_fontsize is None:
                with QSignalBlocker(self.legend_font_size):
                    self.legend_font_size.setValue(legend_font)
            spacing, bounds = spacings[self._primary_unit], axes[self._primary_unit]
            if options.spacing is None:
                with QSignalBlocker(self.spacing):
                    self.spacing.setValue(spacing)
            if options.auto_y:
                for field, value in zip((self.ymin, self.ymax), bounds):
                    with QSignalBlocker(field):
                        field.setValue(value)
            if options.y_tick_interval is None:
                with QSignalBlocker(self.y_interval):
                    self.y_interval.setValue(suggested_y_interval(*bounds))
            if len(axes) > 1:
                if options.right_spacing is None:
                    with QSignalBlocker(self.right_spacing):
                        self.right_spacing.setValue(spacings['mAU'])
                if options.right_auto_y:
                    for field, value in zip((self.right_ymin, self.right_ymax), axes['mAU']):
                        with QSignalBlocker(field):
                            field.setValue(value)
                if options.right_y_tick_interval is None:
                    with QSignalBlocker(self.right_y_interval):
                        self.right_y_interval.setValue(suggested_y_interval(*axes['mAU']))
            self.summary.setText(f"{len(traces)} traces · Overlay" if options.layout == 'overlay' else
                                f"{len(traces)} traces · spacing {spacing:.6g} {self._primary_unit}" +
                                 (f" / {spacings['mAU']:.6g} mAU" if len(axes) > 1 else ''))
            self.error.clear()
            self.preview.show()
            self.save_button.setEnabled(True)
        except (ValueError, OverflowError, ImportError, OSError) as exc:
            # Matplotlib loads with the first preview, so its import and setup errors show here.
            self.svg_bytes = b""
            self._rendered = None
            self.preview.hide()
            self.error.setText(str(exc))
            self.summary.clear()
            self.save_button.setEnabled(False)

    def _update_axis_controls(self, traces):
        units = {trace.unit for trace in traces}
        primary = 'counts' if 'counts' in units or not units else 'mAU'
        if primary != self._primary_unit:
            self.auto_y.setChecked(True)
            self.auto_spacing.setChecked(True)
            self._y_interval_edited = False
        self._primary_unit = primary
        stacked = self.layout_choice.currentText() == 'Stacked'
        for field in (self.auto_spacing, self.right_auto_spacing, self.offset):
            field.setEnabled(stacked)
        self.spacing.setEnabled(stacked and not self.auto_spacing.isChecked())
        self.right_spacing.setEnabled(stacked and not self.right_auto_spacing.isChecked())
        xname, xunit = ('Wavelength', 'nm') if self.spectrum_kind == 'dad' else ('m/z', 'm/z') if self.spectrum_kind else ('RT', 'min')
        for field, label in ((self.xmin, f'{xname} minimum ({xunit})'), (self.xmax, f'{xname} maximum ({xunit})'),
                             (self.x_interval, f'{xname} tick interval ({xunit})'), (self.offset, f'Successive {xname} offset ({xunit})'),
                             (self.grid_interval, f'Grid spacing ({xunit})'), (self.spacing, f'Stack spacing ({primary})'),
                             (self.y_interval, f'Y tick interval ({primary})')):
            self.form.labelForField(field).setText(label)
            field.setAccessibleName(label)
        self.x_interval.setToolTip(f'Ticks repeat from zero at this interval in {xunit}.')
        self.grid_interval.setToolTip(f'Spacing in {xunit} at the bottom baseline.')
        for field in (self.right_auto_y, self.right_ymin, self.right_ymax, self.right_auto_spacing, self.right_spacing, self.right_y_interval):
            self.form.setRowVisible(field, len(units) > 1)

    def save_svg(self):
        self.update_preview()
        if not self.svg_bytes:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save SVG", self.default_path, "SVG (*.svg)")
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path += ".svg"
        try:
            Path(path).write_bytes(self.svg_bytes)
        except OSError as exc:
            QMessageBox.critical(self, "Save SVG", str(exc))
            return
        self.saved_path = path
        self.default_path = path
        self.summary.setText(f"Saved {Path(path).name}")
