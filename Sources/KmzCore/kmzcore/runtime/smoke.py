"""Explicit development-only frozen GUI smoke mode, using synthetic inputs."""
from pathlib import Path
import json
import os
import platform
import resource
import sys
import time
import traceback

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QFileOpenEvent
from kmzcore.runtime import resource_path
from kmzcore.runtime.synthetic import write_fixture


def run_smoke(app, window, key, destination):
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    external = '--smoke-external' in sys.argv
    a, b = output / 'Synthetic Ω one.mzML', output / 'Synthetic two.mzML'
    if not external:
        write_fixture(a)
        write_fixture(b)
        # Delivered before attach: exercises startup queue without argv emulation.
        app.sendEvent(app, QFileOpenEvent(str(a)))
    state = {'step': 0}
    started = time.monotonic()
    app.setQuitOnLastWindowClosed(False)
    timer = QTimer(window)
    timer.setInterval(20)

    def tick():
        try:
            if time.monotonic() - started > 90:
                raise TimeoutError(f'Smoke timed out at step {state["step"]}')
            if state['step'] == 0:
                if window.runtime.job is not None or str(a) not in window.runtime.opened:
                    return
                assert resource_path(window.runtime.identity.manual).is_file()
                assert resource_path('resources/THIRD_PARTY_NOTICES.txt').is_file()
                if getattr(sys, 'frozen', False):
                    assert resource_path('resources/THIRD_PARTY_LICENSES/CPython-LICENSE.txt').is_file()
                    assert resource_path('resources/LICENSE').is_file()
                    assert resource_path('resources/THIRD_PARTY_LICENSES/Qt-PySide-license-terms/SOURCE.txt').is_file()
                    bundle = Path(sys.executable).parents[2].resolve()
                    assert all(Path(p).resolve().is_relative_to(bundle) for p in sys.path)
                    if key == 'ms1':
                        from kmzviewer.mzml_direct import ACCELERATED
                        assert ACCELERATED, 'Frozen KmzViewer is missing its native mzML decoder'
                    from kmzcore import psi_ms
                    assert psi_ms.vocabulary().version == psi_ms.VERSION  # Bundled, parseable, pinned.
                    # Bundled versions (and pyteomics' copied metadata) engage the exact fast paths.
                    from kmzcore.qualified import qualified
                    names = ('Qt', 'PySide6', 'pyqtgraph') + (('numpy', 'pyteomics', 'ms-entropy') if key == 'ms1' else ())
                    assert qualified(*names), 'Frozen viewer dependencies are not the qualified versions'
                if key == 'ms1':
                    loaded = next(iter(window.session.files.values()))
                    assert loaded.scans.tic.tolist() == [40., 50.]
                    window.extract_request('100', 10, 'all', False, [loaded.file_id])
                else:
                    window.precursor.setText('300')
                    window._apply_filters()
                    assert window.model.rowCount() == 1
                (output / 'ready.json').write_text(json.dumps({'request': str(b)}))
                if not external:
                    # MS1 receives this while extraction is busy; MS2 warm open.
                    app.sendEvent(app, QFileOpenEvent(str(b)))
                    window.open_paths([str(a), str(b), QUrl.fromLocalFile(str(b))])
                state['step'] = 1
            elif state['step'] == 1:
                if window.runtime.job is not None or window.runtime.pending or str(b) not in window.runtime.opened:
                    return
                assert window.runtime.opened == {str(a), str(b)}
                if key == 'ms1':
                    loaded = next(iter(window.session.files.values()))
                    assert loaded.traces[1].y.tolist() == [5., 8.]
                    if not state.get('bpc_requested'):
                        window.extract_bpc_request('positive', [loaded.file_id])
                        state['bpc_requested'] = True
                        return
                    assert loaded.traces[2].kind == 'bpc' and loaded.traces[2].y.tolist() == [5., 8.]
                    if not state.get('filter_requested'):
                        window.apply_mz_filter(200., 200.)
                        state['filter_requested'] = True
                        return
                    assert window._display_trace(loaded.traces[0]).y.tolist() == [2., 3.]
                    assert window._display_trace(loaded.traces[1]).y.tolist() == [0., 0.]
                    assert window._display_trace(loaded.traces[2]).y.tolist() == [2., 3.]
                    from kmzviewer.export_svg import ExportOptions
                    filtered_svg = output / 'filtered.svg'
                    window.export_trace_ids([loaded.traces[0].trace_id], str(filtered_svg), ExportOptions(xmin=1., xmax=2.))
                    assert 'm/z 200–200' in filtered_svg.read_text()
                    window.clear_mz_filter()
                    assert window.plots._rows[0][4].tolist() == [40., 50.]
                    window.apply_mz_filter(200., 200.)
                    assert not window._busy and 'cached' in window.mz_filter.status.text()
                    window.clear_mz_filter()
                    window.walk_trace_id = loaded.traces[0].trace_id
                    window.walk_i = 0
                    window.walk_mode = True
                    window.walk_by(0)
                else:
                    window.table.setCurrentIndex(window.proxy.index(0, 0))
                    window.open_selected()
                    assert 'synthetic-2' in window.spectrum.caption.text()
                    assert 'H2O' in window.spectrum.formula_texts()
                state['step'] = 2
            elif state['step'] == 2:
                if key == 'ms1':
                    if 'loading' in window.spectrum.caption.text():
                        return
                    if not state.get('comparison'):
                        assert 'synthetic-0' in window.spectrum.caption.text()
                        second = list(window.session.files.values())[1]
                        window.toggle_spectrum(second.file_id)
                        window.spectrum.plot.setXRange(90, 160, padding=0)
                        state['comparison'] = True
                        return
                    assert len(window.spectrum._frames) == 2
                    assert all(frame.rt == 1. for frame in window.spectrum._frames)
                    for mode in ('overlay', 'stacked'):
                        window.spectrum.set_mode(mode)
                        assert all(abs(plot.viewRange()[0][0] - 90) < 1e-6 for plot in window.spectrum._plots())
                    assert all(button.isChecked() for button in window._spectrum_eyes.values())
                    # Bulk navigator operations and order must survive freezing too.
                    from kmzviewer.gui.dialogs import ExtractDialog
                    from PySide6.QtCore import Qt
                    loaded = next(iter(window.session.files.values()))
                    selected = {loaded.traces[1].trace_id}
                    window.tree.topLevelItem(0).child(1).setSelected(True)
                    assert window._selected_file_ids() == [loaded.file_id]
                    extract = ExtractDialog([(f.file_id, f.name) for f in window.session.files.values()],
                                            window, selected_file_ids=window._selected_file_ids())
                    assert extract.checked_file_ids() == [loaded.file_id]
                    extract.close()
                    window.set_trace_colors(selected, '#b12345')
                    assert loaded.traces[0].color == loaded.color and loaded.traces[1].color == '#b12345'
                    window.set_trace_visibility(selected, only=True)
                    assert [t.trace_id for _, t in window.session.ordered_traces(True)] == list(selected)
                    window.set_trace_visibility(selected)
                    assert not window.session.ordered_traces(True)
                    for i in range(window.tree.topLevelItemCount()):
                        window.tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)
                    window.move_files([loaded.file_id], len(window.session.files))
                    assert list(window.session.files)[-1] == loaded.file_id
                    assert window.spectrum._frames[-1].file_id == loaded.file_id
                    window.sort_files()
                    assert window.tree.file_ids() == list(window.session.files)
                    source_order = [frame.file_id for frame in window.spectrum._frames]
                    window.set_grouping('type')
                    assert window.tree.topLevelItem(0).text(0) == 'TIC MS1'
                    assert [row[0] for row in window.plots._rows] == [t.trace_id for _, t in window.session.ordered_traces(True)]
                    assert [frame.file_id for frame in window.spectrum._frames] == source_order
                    window.set_grouping('file')
                    from kmzviewer.export_svg import ExportOptions, svg_is_vector
                    ids = [trace.trace_id for f in window.session.files.values() for trace in f.traces]
                    window.export_trace_ids(ids, str(output / 'traces.svg'), ExportOptions('stacked', 6.5, 4.5, 1.1, 10, 1, 2, True, None, None, True, True, False))
                    assert svg_is_vector(output / 'traces.svg')
                    assert '<text' in (output / 'traces.svg').read_text()
                    from kmzviewer.export_svg import ExportTrace
                    from kmzviewer.gui.svg_preview import ExportDialog
                    traces = {trace.trace_id: ExportTrace(f.name, trace.label, trace.color, trace.rt, trace.y)
                              for f, trace in window.session.ordered_traces()}
                    preview = ExportDialog([(tid, f"{trace.filename} — {trace.label}", True) for tid, trace in traces.items()],
                                           (1, 2), (1, 2), trace_data=traces, parent=window)
                    assert preview.preview.renderer().isValid() and preview.svg_bytes
                    first = preview.selected_trace_ids()[0]
                    preview.table.move_row(0, preview.table.rowCount())
                    assert preview.selected_trace_ids()[-1] == first
                    preview.offset.setValue(.1)
                    preview.grid.setChecked(True)
                    preview.line_width.setValue(2.5)
                    preview.update_preview()
                    assert preview.save_button.isEnabled()
                    assert b"clipPath" not in preview.svg_bytes and b"<g " not in preview.svg_bytes
                    assert b"Ion Counts" in preview.svg_bytes and b"stroke-dasharray" in preview.svg_bytes
                    assert b"ppm" not in preview.svg_bytes and b".mzML" not in preview.svg_bytes
                    (output / 'preview.svg').write_bytes(preview.svg_bytes)
                    preview.show()
                    preview.grab().save(str(output / 'svg-preview.png'))
                    preview.close()

                # Deliver real key events through Qt's shortcut map after focusing
                # each plot area; exercise the packaged F shortcut independently.
                from PySide6.QtCore import QEvent, Qt
                from PySide6.QtGui import QKeyEvent
                areas = [(window.spectrum.view if key == 'ms1' else window.spectrum.plot,
                          window.spectrum.plot, (110, 120) if key == 'ms1' else (220, 230))]
                if key == 'ms1':
                    areas.insert(0, (window.plots.view, window.plots._plots()[0], (1.1, 1.2)))
                window.activateWindow()
                for widget, plot, zoom in areas:
                    plot.setXRange(*zoom, padding=0)
                    widget.setFocus()
                    app.processEvents()
                    active = (app.applicationState() == Qt.ApplicationState.ApplicationActive
                              and window.isActiveWindow() and widget.hasFocus())
                    app.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier))
                    app.sendEvent(widget, QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier))
                    if active:
                        assert plot.viewRange()[0][0] < zoom[0] and plot.viewRange()[0][1] > zoom[1], (key, zoom, plot.viewRange()[0])
                        state.setdefault('f_checks', set()).add('F active full view')
                    else:
                        # The launcher smoke starts sibling apps concurrently.
                        # Their background windows must not consume shortcuts.
                        assert all(abs(a-b) < 1e-6 for a, b in zip(plot.viewRange()[0], zoom)), (key, zoom, plot.viewRange()[0])
                        state.setdefault('f_checks', set()).add('F inactive window ignored')

                window.grab().save(str(output / f'{key}.png'))
                from PySide6.QtCore import Qt
                app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
                state['dark_since'] = time.monotonic()
                state['step'] = 3
            elif state['step'] == 3:
                if time.monotonic() - state['dark_since'] < .25:
                    return
                if key == 'ms1' and not state.get('gnps_done'):
                    from kmzviewer.gui.gnps_smoke import step as gnps_step
                    state['gnps_done'] = gnps_step(window, a, output, state.setdefault('gnps', {}))
                    if not state['gnps_done']:
                        return
                if key == 'ms1' and not state.get('spectral_done'):
                    from kmzviewer.gui.spectral_smoke import step as spectral_step
                    state['spectral_done'] = spectral_step(window, a, output, state.setdefault('spectral', {}))
                    if not state['spectral_done']:
                        return
                if key == 'ms1' and not state.get('dad_done'):
                    if not state.get('dad_started'):
                        assert window.dad_dock.isHidden()
                        optical = output / 'Synthetic optical.mzML'
                        write_fixture(optical, include_dad=True)
                        assert not window.load_paths_sync([str(optical)])
                        loaded = list(window.session.files.values())[-1]
                        assert [t.kind for t in loaded.traces] == ['tic'] and window.dad_dock.isHidden()
                        from kmzviewer.dad import WavelengthBand, WavelengthExtraction
                        window.extract_wavelength_sync([WavelengthExtraction(), WavelengthExtraction(WavelengthBand(254, 4))], [loaded.file_id])
                        assert abs(loaded.traces[1].y[0] - 40.) < 1e-10
                        assert abs(loaded.traces[2].y[0] - 43./3) < 1e-10
                        window.inspect_rt(loaded.traces[1].trace_id, 1.1)
                        window.overlay_action.setChecked(True)
                        state['dad_started'] = True
                        return
                    if window._inflight_comparison is not None or window._queued_comparison is not None:
                        return
                    assert window.dad_dock.isVisible()
                    assert window.dad_spectrum._frames[0].rt == 1.1
                    assert min(window.dad_spectrum._frames[0].intensity) == -2.
                    assert window.plots.relative_overlay
                    assert window.plots._right_view is None
                    window.match_action.setChecked(True)
                    assert window.plots._right_view is not None
                    window.plots._right_view.setXRange(1.2, 1.8, padding=0)
                    assert all(abs(a-b) < 1e-6 for a, b in zip(window.plots.current_x_range(), (1.2, 1.8)))
                    assert window.plots._right_view.viewRange()[1][1] > 1
                    window.match_action.setChecked(False)
                    window.dad_spectrum.set_mode('overlay')
                    window.dad_spectrum.plot.setXRange(252, 256, padding=0)
                    assert window.dad_spectrum.relative_overlay
                    curve = next(iter(window.dad_spectrum._curves.values()))
                    visible = curve.yData[(curve.xData >= 252) & (curve.xData <= 256)]
                    assert abs(float(visible.max()) - 100.) < 1e-8
                    for optical in (False, True):
                        preview = window.spectrum_export_dialog(optical)
                        assert preview is not None and preview.svg_bytes, preview.error.text() if preview else 'No spectra'
                        (output / ('dad-spectrum.svg' if optical else 'ms1-spectrum.svg')).write_bytes(preview.svg_bytes)
                        preview.deleteLater()
                    window.grab().save(str(output / 'dad-overlay.png'))
                    # Exercise repeated session actions through the shipped dialog,
                    # including a hidden BPC and a complete optical family.
                    remove_ids = {t.trace_id for _, t in window.session.ordered_traces() if t.kind in {'bpc', 'twc'}}
                    bpc = next(t for _, t in window.session.ordered_traces() if t.kind == 'bpc')
                    window.session.set_visible(bpc.trace_id, False)
                    window._refresh_all()
                    included = (set(window.spectrum_file_ids), set(window.dad_file_ids))
                    files_before = set(window.session.files)
                    from kmzviewer.gui.manage_chromatograms import ManageChromatogramsDialog
                    from PySide6.QtCore import Qt
                    from PySide6.QtWidgets import QApplication
                    timer.stop()
                    def choose_removals():
                        dialog = QApplication.activeModalWidget()
                        try:
                            assert isinstance(dialog, ManageChromatogramsDialog)
                            assert not dialog.selected_trace_ids()
                            for i in range(dialog.tree.topLevelItemCount()):
                                item = dialog.tree.topLevelItem(i)
                                if item.text(0).startswith(('BPC —', 'TWC —')):
                                    item.setCheckState(0, Qt.CheckState.Checked)
                            assert dialog.selected_trace_ids() == remove_ids
                            for action, visible in (("show_plots", True), ("hide_plots", False)):
                                dialog.action_buttons[action].click()
                                assert dialog.isVisible() and dialog.selected_trace_ids() == remove_ids
                                assert all(t.visible == visible for _, t in window.session.ordered_traces() if t.trace_id in remove_ids)
                            for detector, source_ids in (("ms1", included[0]), ("dad", included[1])):
                                for operation in ("hide", "show", "toggle", "toggle"):
                                    dialog.action_buttons[f"{operation}_{detector}"].click()
                                    assert dialog.isVisible() and dialog.selected_trace_ids() == remove_ids
                                # Only a subset of open sources may have been
                                # included before this smoke's batch commands.
                                actual = window.spectrum_file_ids if detector == "ms1" else window.dad_file_ids
                                window.set_spectrum_visibility(actual - source_ids, detector=detector, visible=False)
                                window.set_spectrum_visibility(source_ids, detector=detector, visible=True)
                                actual = window.spectrum_file_ids if detector == "ms1" else window.dad_file_ids
                                assert actual == source_ids
                            dialog.remove_button.click()
                            assert dialog.isVisible() and not dialog.selected_trace_ids()
                            dialog.reject()
                        except Exception:
                            state['removal_error'] = traceback.format_exc()
                            if dialog is not None:
                                dialog.reject()
                    QTimer.singleShot(0, choose_removals)
                    window.manage_chromatograms_button.click()
                    timer.start()
                    assert not state.get('removal_error'), state.get('removal_error')
                    assert not remove_ids & {t.trace_id for _, t in window.session.ordered_traces()}
                    assert set(window.session.files) == files_before
                    assert (window.spectrum_file_ids, window.dad_file_ids) == included
                    window.set_grouping('type')
                    window.tree.clearSelection()
                    window.tree.topLevelItem(0).setSelected(True)  # TIC group represents every source.
                    assert len(window._selected_file_ids()) == len(window.session.files)
                    window.close_selected_files()
                    assert not window.session.files and not window.spectrum._frames and not window.dad_spectrum._frames
                    state['dad_done'] = True
                window.grab().save(str(output / f'{key}-dark.png'))
                state['step'] = 4
                # Uses the same application-level Quit event as native termination.
                from PySide6.QtCore import QEvent
                app.sendEvent(app, QEvent(QEvent.Type.Quit))
            elif state['step'] == 4 and window.runtime.closed:
                timer.stop()
                result = {'ok': True, 'viewer': key, 'frozen': bool(getattr(sys, 'frozen', False)),
                          'platform': platform.platform(), 'python': sys.version,
                          'seconds': round(time.monotonic() - started, 3),
                          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'qt_platform': app.platformName(), 'sys_path': sys.path,
                          'gnps_offline_smoke': state.get('gnps_done') if key == 'ms1' else None,
                          'spectral_offline_smoke': state.get('spectral_done') if key == 'ms1' else None,
                          'mplconfigdir': os.environ.get('MPLCONFIGDIR'),
                          'document_delivery': 'LaunchServices cold/warm' if external else 'synthetic QFileOpenEvent and busy queue',
                          'checks': ['manual', 'notices', 'two documents', 'deduplication', 'spectrum', *sorted(state.get('f_checks', [])), 'MS1 comparison/linked zoom/percent overlay/eyes/SVG/TIC/BPC/EIC/mz filter/cached reapply/filtered SVG/Clear/grouped removal/hidden traces/type grouping/multi-close/file ordering/DAD opt-in/TWC/EWC/dual axes/autoscale/spectrum SVG' if key == 'ms1' else 'precursor filter/H2O hypothesis', 'Quit/reader cleanup']}
                (output / 'result.json').write_text(json.dumps(result, indent=2))
                app.exit(0)
        except Exception:
            timer.stop()
            (output / 'result.json').write_text(json.dumps({'ok': False, 'traceback': traceback.format_exc()}, indent=2))
            window.close()
            # Keep the loop alive until asynchronous shutdown is complete.
            failure = QTimer(window)
            failure.timeout.connect(lambda: app.exit(1) if window.runtime.closed else None)
            failure.start(20)
            window._smoke_failure_timer = failure
    timer.timeout.connect(tick)
    timer.start()
    window._smoke_timer = timer
