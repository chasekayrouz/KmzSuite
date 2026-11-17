"""Preview/save identity, real table events and export-only editing."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog, QFileDialog
from kmzviewer.export_svg import ExportTrace
from kmzviewer.gui.svg_preview import ExportDialog


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def make_dialog(count=3):
    data={i:ExportTrace(f'Sample_{i}.sp.mzML','EIC 273.1559 ±10 ppm',
                       ('#0072b2','#d55e00','#009e73')[i%3],
                       np.array([0.,1.,2.]),np.array([0.,10.*(i+1),0.])) for i in range(count)}
    return ExportDialog([(i,f'{t.filename} — {t.label}',True) for i,t in data.items()],
                        (0,2),trace_data=data)


def test_defaults_global_changes_and_exact_save(app, tmp_path, monkeypatch):
    trace=ExportTrace('Example.mzML','TIC','#0072b2',np.array([0.,1.,2.]),np.array([0.,10.,0.]))
    dialog=ExportDialog([(1,'Example.mzML — TIC',True)],(.5,1.5),(0,2),trace_data={1:trace},default_path=tmp_path/'Example.svg')
    try:
        assert (dialog.xmin.value(),dialog.xmax.value()) == (.5,1.5)
        assert dialog.line_width.value() == 2
        assert dialog.x_interval.value() == 1
        assert dialog.spacing.value() == 11
        assert dialog.preview.renderer().isValid()
        assert dialog.table.columnCount() == 2
        original=dialog.svg_bytes
        dialog.line_width.setValue(4)
        dialog.shading.setChecked(False)
        dialog.x_interval.setValue(.5)
        dialog.update_preview()
        assert original != dialog.svg_bytes
        assert b'stroke-width="4"' in dialog.svg_bytes
        assert b'fill-opacity' not in dialog.svg_bytes
        assert trace.color == '#0072b2'
        target=tmp_path/'export.svg'
        def choose(*args):
            assert args[2] == str(tmp_path/'Example.svg')
            return str(target),'SVG (*.svg)'
        monkeypatch.setattr(QFileDialog,'getSaveFileName',choose)
        dialog.save_svg()
        assert target.read_bytes() == dialog.svg_bytes
        dialog.table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        dialog.update_preview()
        assert not dialog.svg_bytes and not dialog.save_button.isEnabled()
        assert 'at least one' in dialog.error.text()
    finally:
        dialog.close()


def test_selection_bottom_fixed_offset_and_invalid_recovery(app):
    dialog=make_dialog(2)
    try:
        assert dialog.spacing.value() == 22
        dialog.offset.setValue(3)
        dialog.update_preview()
        assert dialog.spacing.value() == 22  # Bottom remains at its original RT.
        dialog.table.item(1,0).setCheckState(Qt.CheckState.Unchecked)
        dialog.update_preview()
        assert dialog.spacing.value() == 11  # Remaining trace becomes the fixed bottom.
        dialog.auto_spacing.setChecked(False)
        dialog.spacing.setValue(7)
        dialog.update_preview()
        assert dialog.export_options().spacing == 7
        dialog.xmax.setValue(-1)
        dialog.update_preview()
        assert not dialog.save_button.isEnabled()
        dialog.xmax.setValue(2)
        dialog.update_preview()
        assert dialog.save_button.isEnabled()
        automatic_bounds=(dialog.ymin.value(),dialog.ymax.value())
        dialog.auto_y.setChecked(False)
        assert (dialog.export_options().ymin,dialog.export_options().ymax) == automatic_bounds
    finally:
        dialog.close()


def test_drag_drop_preserves_identity_color_checks_and_keyboard_order(app,monkeypatch):
    dialog=make_dialog()
    dialog.show()
    app.processEvents()
    table=dialog.table
    try:
        table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        monkeypatch.setattr(QColorDialog,'getColor',lambda *args:QColor('#123456'))
        QTest.mouseClick(table.viewport(),Qt.MouseButton.LeftButton,pos=table.visualItemRect(table.item(0,1)).center())
        assert table.item(0,1).text() == '#123456'
        mime=table.mime_for_row(0)
        point=table.visualItemRect(table.item(2,0)).bottomLeft()+QPoint(80,-1)
        enter=QDragEnterEvent(point,Qt.DropAction.MoveAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(table.viewport(),enter)
        move=QDragMoveEvent(point,Qt.DropAction.MoveAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(table.viewport(),move)
        assert move.isAccepted()
        drop=QDropEvent(QPointF(point),Qt.DropAction.MoveAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(table.viewport(),drop)
        assert drop.isAccepted()
        assert table.item(2,0).data(Qt.ItemDataRole.UserRole) == 0
        assert table.item(2,0).checkState() == Qt.CheckState.Unchecked
        assert table.item(2,1).text() == '#123456'
        assert dialog.selected_trace_ids() == [1,2]
        QTest.keyClick(table,Qt.Key.Key_Up,Qt.KeyboardModifier.AltModifier)
        assert table.item(1,0).data(Qt.ItemDataRole.UserRole) == 0
        QTest.mouseClick(table.viewport(),Qt.MouseButton.LeftButton,
                         pos=table.visualItemRect(table.item(1,0)).topLeft()+QPoint(10,14))
        assert table.item(1,0).checkState() == Qt.CheckState.Checked
        assert dialog.selected_trace_ids() == [1,0,2]
        dialog.offset.setValue(1)
        dialog.update_preview()
        assert dialog.selected_traces()[1].color == '#123456'
        assert dialog._data[0].color == '#0072b2'
        assert dialog.table.item(1,0).text().endswith('±10 ppm')
    finally:
        dialog.close()


def test_interval_edit_scientific_counts_and_wheel_safety(app):
    dialog=make_dialog()
    dialog.show()
    app.processEvents()
    try:
        dialog.y_interval.setFocus()
        dialog.y_interval.selectAll()
        QTest.keyClicks(dialog.y_interval,'1e1')
        dialog.update_preview()
        assert dialog.export_options().y_tick_interval == 10
        dialog.offset.setValue(1)
        dialog.update_preview()
        assert dialog.y_interval.value() == 10
        dialog.y_interval.selectAll()
        QTest.keyClicks(dialog.y_interval,'nan')
        dialog.update_preview()
        assert not dialog.save_button.isEnabled()
        dialog.y_interval.selectAll()
        QTest.keyClicks(dialog.y_interval,'5e0')
        dialog.update_preview()
        assert dialog.save_button.isEnabled()
        previous=dialog.line_width.value()
        wheel=QWheelEvent(QPointF(10,10),QPointF(10,10),QPoint(),QPoint(0,120),
                          Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,
                          Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(dialog.line_width,wheel)
        assert dialog.line_width.value() == previous
    finally:
        dialog.close()


def test_export_uses_last_input_folder_and_current_view(app,tmp_path,monkeypatch):
    from kmzviewer.gui.app import MainWindow
    from Tests.Viewer.mzml_factory import write_mzml
    paths=[tmp_path/'first.mzML',tmp_path/'second.mzML']
    for index,path in enumerate(paths):
        write_mzml(path,[{'mz':[100.], 'intensity':[10.], 'rt':1., 'tic':10., 'polarity':1},
                         {'mz':[100.], 'intensity':[20.], 'rt':2.+index*3, 'tic':20., 'polarity':1}])
    window=MainWindow()
    try:
        assert not window.load_paths_sync([str(p) for p in paths])
        last=list(window.session.files.values())[-1]
        window.session.set_visible(last.traces[0].trace_id,False)
        window.plots.set_x_range(1.25,1.75)
        seen=[]
        def inspect(dialog):
            assert dialog.default_path == str(paths[-1].with_suffix('.svg'))
            assert (dialog.xmin.value(),dialog.xmax.value()) == (1.25,1.75)
            assert dialog.table.rowCount() == 2
            assert dialog.table.item(1,0).checkState() == Qt.CheckState.Unchecked
            assert dialog.svg_bytes
            seen.append(True)
            return 0
        monkeypatch.setattr(ExportDialog,'exec',inspect)
        window.open_export_dialog(single=False)
        assert seen == [True]
    finally:
        window.close()
        app.processEvents()


def test_long_table_drag_autoscroll_and_cancel(app):
    from PySide6.QtGui import QDragLeaveEvent
    dialog=make_dialog(30)
    dialog.legend.setChecked(False)
    dialog.update_preview()
    dialog.show()
    app.processEvents()
    table=dialog.table
    try:
        mime=table.mime_for_row(0)
        point=QPoint(80,table.viewport().height()-4)
        enter=QDragEnterEvent(point,Qt.DropAction.MoveAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(table.viewport(),enter)
        move=QDragMoveEvent(point,Qt.DropAction.MoveAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(table.viewport(),move)
        QTest.qWait(260)
        assert table.verticalScrollBar().value() >= 2
        QApplication.sendEvent(table.viewport(),QDragLeaveEvent())
        assert not table._scroll_timer.isActive()
        assert dialog.selected_trace_ids() == list(range(30))
    finally:
        dialog.close()


def test_legend_font_fits_full_names_then_retains_user_size(app):
    import xml.etree.ElementTree as ET
    trace = ExportTrace('Long_sample_description_' * 5 + '.mzML', 'TIC', '#0072b2',
                        np.array([0., 1.]), np.array([0., 10.]))
    dialog = ExportDialog([(1, trace.filename + ' — TIC', True)], (0, 1), trace_data={1: trace})
    try:
        initial = dialog.legend_font_size.value()
        assert .1 <= initial < 10
        assert dialog.export_options().legend_fontsize is None
        dialog.width.setValue(9)
        dialog.update_preview()
        assert dialog.legend_font_size.value() > initial
        dialog.legend_font_size.setValue(initial / 2)
        dialog.update_preview()
        chosen = dialog.legend_font_size.value()
        assert dialog.export_options().legend_fontsize == chosen
        dialog.font_size.setValue(14)
        dialog.width.setValue(10)
        dialog.update_preview()
        assert dialog.legend_font_size.value() == chosen
        labels = [n for n in ET.fromstring(dialog.svg_bytes) if n.get('data-legend')]
        assert float(labels[0].get('font-size')) == chosen
        dialog.legend_font_size.setValue(72)
        dialog.update_preview()
        assert not dialog.save_button.isEnabled()
        dialog.legend_font_size.setValue(chosen)
        dialog.update_preview()
        assert dialog.save_button.isEnabled()
        dialog.legend.setChecked(False)
        dialog.update_preview()
        assert not any(n.get('data-legend') for n in ET.fromstring(dialog.svg_bytes))
    finally:
        dialog.close()
