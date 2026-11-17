from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from PySide6.QtCore import QCoreApplication,QEvent,QEventLoop,QThread,QTimer,Qt
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox

from kmzcomet.engine import analyze
from kmzcomet.history import load_analysis,payload,save_analysis
from kmzcomet.model import Settings
from kmzcomet.sources import InputSource
from kmzcomet_app.app import MainWindow
from kmzcomet_app.table import FeatureFilter,MatrixModel
from kmzcomet_app.workers import Completed,Worker,analysis_worker
from scripts.benchmarks.comet_fixtures import Truth,write_run


def until(predicate,timeout=10000):
    if predicate():return
    loop=QEventLoop();tick=QTimer();tick.setInterval(2)
    tick.timeout.connect(lambda:loop.quit() if predicate() else None)
    deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
    tick.start();deadline.start(timeout);loop.exec();tick.stop()
    assert predicate(),'GUI event-loop deadline expired'


@pytest.fixture(scope='session')
def qt_application():
    # Keep one application alive across the mixed CoMet/viewer test process.
    return QApplication.instance() or QApplication([])


@pytest.fixture
def app(tmp_path,monkeypatch,qt_application):
    monkeypatch.setenv('KMZ_LOCAL_STATE_ROOT',str(tmp_path/'state'))
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.Yes)
    app=qt_application
    previous=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    yield app
    for widget in app.topLevelWidgets():
        if isinstance(widget,MainWindow):
            widget.close();until(lambda:widget.runtime.closed);widget.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    app.processEvents();app.setQuitOnLastWindowClosed(previous)


@pytest.fixture
def result(tmp_path):
    truth=[Truth('a',200.,1.,(9000.,600.)),Truth('b',300.,1.5,(8000.,0.))]
    paths=[write_run(tmp_path/f'{i}.mzML',truth,sample=i,noise=0.,rt_stop=2.) for i in range(2)]
    return analyze(paths,Settings(alignment='none'))


def test_numeric_sort_missing_values_and_zero(app,result):
    result.measurements[0][0].height=None;result.measurements[0][1].height=0.
    model=MatrixModel(result);proxy=FeatureFilter();proxy.setSourceModel(model)
    for order in (Qt.SortOrder.AscendingOrder,Qt.SortOrder.DescendingOrder):
        proxy.sort(4,order)
        assert proxy.index(0,4).data(Qt.ItemDataRole.UserRole)==0.
        assert proxy.index(1,4).data(Qt.ItemDataRole.UserRole) is None
    proxy.set_filters('',(250.,None,None,None));assert proxy.rowCount()==1
    assert proxy.index(0,1).data(Qt.ItemDataRole.UserRole)==300.


def test_analysis_autosave_and_reopen_without_inputs(app,result,tmp_path,monkeypatch):
    from kmzcomet_app import workers
    monkeypatch.setattr(workers,'analyze',lambda *args,**kwargs:result)
    window=MainWindow(history=tmp_path/'history');window.show()
    window.inputs=[InputSource(s.path) for s in result.samples]
    threads=[];original=window.model.set_result
    def set_result(value):threads.append(QThread.currentThread());original(value)
    monkeypatch.setattr(window.model,'set_result',set_result)
    try:
        window.start_analysis();until(lambda:not window.busy)
        assert threads==[app.thread()] and window.result is result
        assert window.saved_path.is_file() and payload(load_analysis(window.saved_path))==payload(result)
        assert window.model.rowCount()==2 and window.model.columnCount()==7
        assert all(window.table.columnWidth(i)>80 for i in (5,6))
        assert all(window.table.frozen.isColumnHidden(i) for i in (5,6))
        saved=window.saved_path
        for sample in result.samples:sample.path.unlink()
        window.open_paths([saved]);until(lambda:not window.busy)
        assert payload(window.result)==payload(result)
        assert window.analysis_settings==result.settings
    finally:window.close();until(lambda:window.runtime.closed)


def test_save_failure_keeps_result_and_retry_commits(app,result,tmp_path,monkeypatch):
    from kmzcomet_app import workers
    monkeypatch.setattr(workers,'analyze',lambda *args,**kwargs:result)
    original=workers.save_analysis
    def fail(*args,**kwargs):raise OSError('disk full')
    monkeypatch.setattr(workers,'save_analysis',fail)
    window=MainWindow(history=tmp_path/'history');window.inputs=[InputSource(s.path) for s in result.samples]
    try:
        window.start_analysis();until(lambda:not window.busy)
        assert window.result is result and window.saved_path is None
        assert not list(window.history.glob('*.kmzcomet'))
        assert window.retry_action.isVisible()
        monkeypatch.setattr(QMessageBox,'question',lambda *args:QMessageBox.StandardButton.No)
        window.start_analysis()
        assert not window.busy and window.result is result and window.saved_path is None
        monkeypatch.setattr(workers,'save_analysis',original)
        window.retry_save();until(lambda:not window.busy)
        assert window.saved_path.is_file()
    finally:window.close();until(lambda:window.runtime.closed)


@pytest.mark.parametrize('close',[False,True])
def test_cancel_and_close_keep_thread_alive_until_decode_returns(app,tmp_path,close):
    entered,release=Event(),Event()
    def operation(worker):
        entered.set();assert release.wait(5);worker.checkpoint()
    worker=Worker(operation);window=MainWindow(history=tmp_path/'history');window.show()
    try:
        window._start(worker,lambda _:pytest.fail('Cancelled job delivered a result'),'analysis')
        until(entered.is_set);thread=window._thread
        if close:window.close();assert window.isVisible() and not window.runtime.closed
        else:window.cancel()
        assert window._thread is thread and thread.isRunning()
        heartbeat=[];QTimer.singleShot(0,lambda:heartbeat.append(True));until(lambda:bool(heartbeat))
        release.set();until(lambda:not window.busy)
        if close:until(lambda:window.runtime.closed)
    finally:
        release.set();window.close();until(lambda:window.runtime.closed)


def test_worker_teardown_never_waits_for_the_gil_under_a_connection_lock(tmp_path):
    from Tests.worker_teardown import run_owner
    process=run_owner('comet',tmp_path/'history')
    assert process.returncode==0,process.stdout+process.stderr


def test_finishing_worker_never_takes_a_connection_lock_while_the_gui_destroys_objects(tmp_path):
    from Tests.worker_teardown import MIRROR,run_owner
    process=run_owner('comet',tmp_path/'history',MIRROR)
    assert process.returncode==0,process.stdout+process.stderr


def test_exports_ignore_transient_table_filters(app,result,tmp_path,monkeypatch):
    import csv
    saved=save_analysis(result,tmp_path/'history')
    window=MainWindow(history=tmp_path/'history');window._completed(Completed(result,saved))
    window.bounds[0].setText('250');assert window.proxy.rowCount()==1
    path=tmp_path/'all.csv'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args,**kwargs:(str(path),'CSV'))
    try:
        window.export('csv');until(lambda:not window.busy)
        with path.open(encoding='utf-8-sig') as handle:rows=list(csv.reader(handle))
        assert len(rows)==3 and rows[1][0]==result.features[0].id
    finally:window.close();until(lambda:window.runtime.closed)


def test_benchmark_preset_is_the_accepted_recipe(app,tmp_path):
    from kmzcomet_app.dialogs import SettingsDialog
    window=MainWindow(history=tmp_path/'history')
    assert window.analysis_settings==Settings(mz_ppm=30,discovery_window=.35,rt_window=.15,min_contrast=2,min_height=100)
    dialog=SettingsDialog(window.analysis_settings,window)
    dialog.accept()
    assert dialog.settings==window.analysis_settings
    dialog.widgets['min_height'].setValue(0)
    dialog.widgets['mz_ppm'].setValue(12.5)
    dialog.accept()
    assert dialog.settings.min_height==0 and dialog.settings.mz_ppm==12.5 and dialog.settings.min_contrast==2
    window.close()


def show_result(app, result, tmp_path):
    window=MainWindow(history=tmp_path/'history')
    window._completed(Completed(result,save_analysis(result,window.history)))
    window.show();window.activateWindow();app.processEvents()
    return window


def click_height(app, window, row, sample):
    from PySide6.QtTest import QTest
    index=window.proxy.index(row,sample+5)
    window.table.scrollTo(index);app.processEvents()
    QTest.mouseClick(window.table.viewport(),Qt.MouseButton.LeftButton,pos=window.table.visualRect(index).center())
    app.processEvents()


def test_eic_click_toggles_and_tristate_row_checkbox(app,result,tmp_path):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    window=show_result(app,result,tmp_path)
    assert window.model.selected_cells==() and not window.table.selectedIndexes()
    assert not window.plot_action.isEnabled()
    assert window.model.headerData(2,Qt.Orientation.Horizontal)=='m/z'
    assert window.proxy.index(0,2).data()=='200.000000'
    click_height(app,window,0,0);click_height(app,window,0,1)
    assert window.model.selected_cells==((0,0),(0,1))
    checkbox=window.proxy.index(0,0)
    assert checkbox.data(Qt.ItemDataRole.CheckStateRole)==Qt.CheckState.Checked
    click_height(app,window,0,0)
    assert window.model.selected_cells==((0,1),)
    assert checkbox.data(Qt.ItemDataRole.CheckStateRole)==Qt.CheckState.PartiallyChecked
    rect=window.table.frozen.visualRect(checkbox)
    QTest.mouseClick(window.table.frozen.viewport(),Qt.MouseButton.LeftButton,pos=rect.topLeft()+QPoint(10,rect.height()//2))
    assert window.model.selected_cells==((0,0),(0,1))
    QTest.mouseClick(window.table.frozen.viewport(),Qt.MouseButton.LeftButton,pos=rect.topLeft()+QPoint(10,rect.height()//2))
    assert window.model.selected_cells==()
    assert not window.plot_action.isEnabled()


def test_eic_picks_survive_sort_filters_and_clear(app,result,tmp_path):
    window=show_result(app,result,tmp_path)
    click_height(app,window,0,0)
    window.table.sortByColumn(2,Qt.SortOrder.DescendingOrder)
    assert window.model.selected_cells==((0,0),)
    window.bounds[0].setText('250')
    assert '1 hidden by filters' in window.selection_note.text() and window.plot_action.isEnabled()
    window.clear_filters()
    click_height(app,window,0,1)
    assert window.model.selected_cells==((0,0),(1,1))
    window.clear_selection.click()
    assert window.model.selected_cells==() and not window.plot_action.isEnabled()


def test_command_e_creates_independent_cached_plot_tabs_and_closes_active_tab(app,result,tmp_path):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest
    from kmzcomet_app.chromatograms import EICTab
    window=show_result(app,result,tmp_path)
    click_height(app,window,0,0);click_height(app,window,0,1)
    QTest.keySequence(window.table,QKeySequence('Ctrl+E'))
    until(lambda:not window.busy and window.tabs.count()==2)
    first=window.tabs.currentWidget()
    assert isinstance(first,EICTab) and len(first.entries)==2
    assert not window.dock.isVisible() and first.batch.files_read==2
    assert first.plots.data_x_range()[0]<.2 and first.plots.data_x_range()[1]>1.8
    window.tabs.setCurrentWidget(window.heights_page);app.processEvents()
    assert window.dock.isVisible()
    click_height(app,window,0,1)
    window.plot_action.trigger();until(lambda:not window.busy and window.tabs.count()==3)
    second=window.tabs.currentWidget()
    assert len(first.entries)==2 and len(second.entries)==1
    assert second.batch.reused==1 and second.batch.files_read==0
    assert second.entries[1].data is first.entries[1].data
    QTest.keySequence(second.plots.view,QKeySequence('Ctrl+W'))
    app.processEvents()
    assert window.tabs.count()==2 and not window.runtime.closed
    assert window.model.selected_cells==((0,0),)
    window.close_plot_tab(window.tabs.indexOf(first));app.processEvents()
    assert window.tabs.currentWidget() is window.heights_page


def test_eic_plot_controls_link_zoom_hide_reorder_and_keep_spectra_disabled(app,result,tmp_path):
    import numpy as np
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest,QSignalSpy
    from kmzviewer.gui.ms1_spectra import SpectrumPanel
    window=show_result(app,result,tmp_path)
    window.model.setData(window.model.index(0,0),Qt.CheckState.Checked,Qt.ItemDataRole.CheckStateRole)
    window.plot_eics();until(lambda:not window.busy and window.tabs.count()==2)
    tab=window.tabs.currentWidget();app.processEvents()
    assert tab.findChildren(SpectrumPanel)==[] and not tab.plots.inspection_enabled
    tab.plots.set_x_range(.8,1.2)
    for plot in tab.plots._plots():np.testing.assert_allclose(plot.viewRange()[0],[.8,1.2],atol=.02)
    tab.match_action.setChecked(True)
    np.testing.assert_allclose(tab.plots._plots()[0].viewRange()[1],tab.plots._plots()[1].viewRange()[1])
    tab.overlay_action.trigger();app.processEvents()
    assert tab.plots._mode=='overlay' and not tab.plots._overlay_plot.legend.isVisible()
    np.testing.assert_allclose(tab.plots.current_x_range(),[.8,1.2],atol=.02)
    tab.toggle_emphasis(1);assert tab.plots._emphasized=={1}
    spy=QSignalSpy(tab.plots.scan_requested);spectra=QSignalSpy(tab.plots.spectrum_added)
    plot=tab.plots._plots()[0]
    point=tab.plots.view.mapFromScene(plot.getViewBox().sceneBoundingRect().center())
    QTest.mouseDClick(tab.plots.view.viewport(),Qt.MouseButton.LeftButton,pos=point)
    app.processEvents()
    assert spy.count()==spectra.count()==0
    tab.stacked_action.trigger();tab.legend.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert len(tab.plots._plots())==1 and len(tab.entries)==2
    tab.legend.item(1).setCheckState(Qt.CheckState.Checked)
    tab.legend.model().moveRow(tab.legend.rootIndex(),0,tab.legend.rootIndex(),2)
    assert [row[0] for row in tab.plots._rows]==[2,1]
    tab.autoscale_action.setChecked(False)
    tab.plots._plots()[0].setYRange(0,1,padding=0)
    tab.plots.view.setFocus();QTest.keySequence(tab.plots.view,QKeySequence('F'));app.processEvents()
    assert tab.plots.current_x_range()[0]<.2 and tab.plots._plots()[0].viewRange()[1][1]>1
    assert not tab.autoscale_action.isChecked() and tab.match_action.isChecked()


def test_unavailable_sources_keep_saved_tables_and_other_eics_usable(app,result,tmp_path):
    window=show_result(app,result,tmp_path)
    result.samples[0].path.unlink()
    window.model.setData(window.model.index(0,0),Qt.CheckState.Checked,Qt.ItemDataRole.CheckStateRole)
    window.plot_eics();until(lambda:not window.busy and window.tabs.count()==2)
    tab=window.tabs.currentWidget()
    assert tab.entries[1].data is None and tab.entries[2].data is not None
    assert '1 unavailable' in tab.note.text() and 'Unavailable:' in tab.legend.item(0).toolTip()
    saved=window.saved_path
    window.open_paths([saved]);until(lambda:not window.busy)
    assert window.tabs.count()==1 and window.model.selected_cells==()
    assert window.result.features and window._eic_cache.nbytes==0


def test_close_during_eic_read_cancels_without_late_plot(app,result,tmp_path,monkeypatch):
    window=show_result(app,result,tmp_path)
    window.model.toggle_cell(0,0)
    entered,release=Event(),Event()
    def extract(cells,*,progress,checkpoint):
        entered.set();assert release.wait(5);checkpoint()
        pytest.fail('Cancellation did not stop plotting')
    monkeypatch.setattr(window._eic_cache,'extract',extract)
    try:
        window.plot_eics();until(entered.is_set)
        window.close();assert window.busy and not window.runtime.closed
        heartbeat=[];QTimer.singleShot(0,lambda:heartbeat.append(True));until(lambda:bool(heartbeat))
        release.set();until(lambda:window.runtime.closed)
        assert window.tabs.count()==1
    finally:
        release.set()


def test_large_eic_stack_legend_scroll_and_tab_local_colors(app,result,tmp_path,monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    from kmzcomet.chromatograms import EICBatch
    from kmzcomet_app.chromatograms import EICTab
    window=show_result(app,result,tmp_path)
    source=window._eic_cache.extract([(0,0),(0,1)]).entries
    batch=EICBatch(tuple(replace(source[i%2],feature_id=f'F{i+1:06d}') for i in range(25)),0,2)
    tab=EICTab(batch,sample_order=['S001','S002'])
    window.tabs.setCurrentIndex(window.tabs.addTab(tab,'25 EICs'));window.resize(1100,700);app.processEvents()
    assert tab.scroll.verticalScrollBar().maximum()>0
    assert tab.legend.verticalScrollBar().maximum()>0
    assert tab.legend.horizontalScrollBar().maximum()==0
    original=tab.colors[2]
    monkeypatch.setattr(QColorDialog,'getColor',lambda *args:QColor('#cc3377'))
    tab.change_color(tab.legend.item(0))
    assert tab.colors[1]=='#cc3377' and tab.colors[2]==original
    assert tab.entries[1].data is source[0].data
    tab.overlay_action.trigger();app.processEvents()
    assert tab.scroll.verticalScrollBar().maximum()==0
    assert len(tab.plots._curves)==25 and not tab.plots._overlay_plot.legend.isVisible()


def test_pca_all_rows_independent_tabs_colors_and_command_click_eics(app,result,tmp_path):
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest
    from kmzcomet_app.chromatograms import EICTab
    from kmzcomet_app.pca import PCATab
    from matplotlib.colors import LogNorm
    window=show_result(app,result,tmp_path)
    click_height(app,window,1,0)
    picks=window.model.selected_cells; original=payload(result)
    window.bounds[0].setText('250')
    window.pca_action.trigger();until(lambda:not window.busy and window.tabs.count()==2)
    tab=window.tabs.currentWidget();assert isinstance(tab,PCATab)
    tab.canvas.draw();app.processEvents()
    assert len(tab.data.points)==2 and len(tab.xyz)==2 and tab.data.rank==1
    assert 'no variation' in tab.axes.yaxis.get_label_text()
    assert isinstance(tab._scatter.norm,LogNorm)
    np.testing.assert_allclose(tab._scatter.get_array(),[sum(c[i].height for c in result.measurements) for i in range(2)])
    assert not window.dock.isVisible()
    # True Qt Command-click reaches the original row, despite proxy filtering.
    position=tab.projected_points()[0]
    QTest.mouseClick(tab.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,
                     QPoint(round(position[0]),round(position[1])))
    until(lambda:not window.busy and window.tabs.count()==3)
    eics=window.tabs.currentWidget();assert isinstance(eics,EICTab)
    assert len(eics.entries)==2
    assert all(entry.feature_id==result.features[0].id for entry in eics.entries.values())
    assert all(entry.data is not None for entry in eics.entries.values())
    assert window.model.selected_cells==picks and payload(result)==original
    window.pca_action.trigger();assert window.tabs.count()==4
    other=window.tabs.currentWidget();assert isinstance(other,PCATab) and other.data is tab.data
    other.color_mode.setCurrentIndex(1);assert isinstance(tab._scatter.norm,LogNorm)
    QTest.keySequence(other.canvas,QKeySequence('Ctrl+W'));app.processEvents()
    assert window.tabs.count()==3 and not window.runtime.closed


def test_pca_reopen_without_originals_missing_and_zero_rows(app,result,tmp_path):
    result.measurements[0][0].height=None
    result.measurements[0][0].status='not_acquired'
    window=show_result(app,result,tmp_path)
    for sample in result.samples:sample.path.unlink()
    window.pca_action.trigger();until(lambda:not window.busy and window.tabs.count()==2)
    tab=window.tabs.currentWidget();tab.canvas.draw()
    assert len(tab.data.points)==1 and tab.data.points[0].row==1
    assert '1 incomplete' in tab.omissions.text() and 'no variation' in tab.summary.text()
    saved=window.saved_path
    window.open_paths([saved]);until(lambda:not window.busy)
    assert window.tabs.count()==1 and window._pca_data is None
    window.pca_action.trigger();until(lambda:not window.busy and window.tabs.count()==2)
    assert len(window.tabs.currentWidget().data.points)==1


def test_pca_rotate_zoom_axis_swap_and_overlapping_feature_choice(app,tmp_path):
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from matplotlib.backend_bases import MouseEvent
    from Tests.CoMet.test_pca import ROWS,matrix_result
    window=MainWindow(history=tmp_path/'history')
    # Data-only tab uses the actual calculation and renderer.
    from kmzcomet.pca import feature_pca
    from kmzcomet_app.pca import PCATab
    tab=PCATab(feature_pca(matrix_result(ROWS)))
    window.tabs.setCurrentIndex(window.tabs.addTab(tab,'PCA'))
    window.show();window.activateWindow();app.processEvents();tab.canvas.draw()
    original=(tab.axes.elev,tab.axes.azim,tab.axes.roll)
    middle=tab.axes.bbox.get_points().mean(axis=0)/tab.canvas.device_pixel_ratio
    position=QPoint(round(middle[0]),round(tab.canvas.height()-middle[1]))
    QTest.mousePress(tab.canvas,Qt.MouseButton.LeftButton,pos=position)
    QTest.mouseMove(tab.canvas,position+QPoint(65,40))
    QTest.mouseRelease(tab.canvas,Qt.MouseButton.LeftButton,pos=position+QPoint(65,40))
    assert (tab.axes.elev,tab.axes.azim,tab.axes.roll)!=original
    limits=tab.axes.get_xlim()
    x,y=tab.canvas.mouseEventCoords(position)
    MouseEvent('scroll_event',tab.canvas,x,y,step=1)._process()
    assert np.ptp(tab.axes.get_xlim())<np.ptp(limits)
    tab.reset_button.click();assert tab.axes.elev==24 and tab.axes.azim==-55
    assert np.ptp(tab.axes.get_xlim())==pytest.approx(np.ptp(tab.axes.get_zlim()))
    tab.axis_boxes[0].setCurrentIndex(3)
    assert tab._axes_selection==[3,1,2] and 'PC4' in tab.axes.xaxis.get_label_text()
    tab.axis_boxes[1].setCurrentIndex(3)
    assert tab._axes_selection==[1,3,2]
    tab.canvas.draw();point=tab.projected_points()[0];rows=[]
    tab.feature_requested.connect(rows.append)
    QTest.mouseClick(tab.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,
                     QPoint(round(point[0]),round(point[1])))
    assert tab._chooser is not None and not rows
    from PySide6.QtWidgets import QListWidget
    listing=tab._chooser.findChild(QListWidget)
    for i in range(listing.count()):
        if listing.item(i).data(Qt.ItemDataRole.UserRole)==1:listing.setCurrentRow(i);break
    tab._chooser.accept();app.processEvents()
    assert rows==[1]


def test_pca_close_during_calculation_waits_and_does_not_publish(app,result,tmp_path,monkeypatch):
    import kmzcomet_app.app as module
    entered,release=Event(),Event()
    def calculate(result,**kwargs):
        entered.set();assert release.wait(5);kwargs['checkpoint']()
    monkeypatch.setattr(module,'feature_pca',calculate)
    window=show_result(app,result,tmp_path)
    try:
        window.pca_action.trigger();until(entered.is_set)
        window.close();assert window.busy and not window.runtime.closed
        release.set();until(lambda:window.runtime.closed)
        assert window.tabs.count()==1 and window._pca_data is None
    finally:release.set()


def test_pca_empty_plot_and_equal_total_color_scales(app,tmp_path):
    from Tests.CoMet.test_pca import matrix_result
    from kmzcomet.pca import feature_pca
    from kmzcomet_app.pca import PCATab
    window=MainWindow(history=tmp_path/'history')
    for rows in ([[None,1.],[0.,0.]], [[1.,2.,3.],[3.,2.,1.]]):
        tab=PCATab(feature_pca(matrix_result(rows)))
        window.tabs.setCurrentIndex(window.tabs.addTab(tab,'PCA'));window.show();app.processEvents()
        tab.canvas.draw();tab.color_mode.setCurrentIndex(1);tab.canvas.draw()
        if tab.data.points:
            assert tab._scatter.norm.vmin<6<tab._scatter.norm.vmax
        else:
            assert tab._scatter is None and not tab.color_axes.get_visible()
