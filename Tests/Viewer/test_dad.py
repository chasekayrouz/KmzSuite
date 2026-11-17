import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from kmzconvert_core.writer import write_mzml
from kmzcore.cancel import Cancelled
from kmzviewer.dad import WavelengthBand, WavelengthExtraction, extract_wavelengths
from kmzviewer.mzml_index import load_ms1
from kmzviewer.spectra import SpectrumReader, ArrayCache
from kmzviewer.gui.app import MainWindow
from kmzviewer.export_svg import ExportTrace, ExportOptions, render_chromatograms, plot_geometry
from Tests.dad_fixtures import acquisition
from Tests.Viewer.test_ms1_comparison import settled, close, click, drag


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def run(tmp_path):
    source = acquisition(tmp_path)
    path = write_mzml(source, tmp_path/'sample.mzML')
    return path, source.dad


def test_index_and_extract_native_times_signed_means_reference_and_cache(run):
    path, source = run
    before = hashlib.sha256(path.read_bytes()).digest()
    scans = load_ms1(path, chunk_size=73)
    assert len(scans) == 16 and len(scans.dad) == 5
    assert not set(scans.spectrum_index) & set(scans.dad.spectrum_index)
    np.testing.assert_array_equal(scans.dad.rt_min, source.rt_min)
    settings = [WavelengthExtraction(), WavelengthExtraction(WavelengthBand(254, 4)),
                WavelengthExtraction(WavelengthBand(254, 4), WavelengthBand(258, 2))]
    cache = ArrayCache()
    traces = extract_wavelengths(str(path), scans.dad, settings, [1,2,3], '#000000', cache=cache)
    np.testing.assert_allclose(traces[0].y, source.total_mau, atol=1e-12)
    np.testing.assert_allclose(traces[1].y, source.absorbance_mau[:,2:7].mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(traces[2].y, source.absorbance_mau[:,2:7].mean(axis=1)-source.absorbance_mau[:,7:10].mean(axis=1), atol=1e-12)
    assert traces[0].y[0] < 0 and all(t.optical for t in traces)
    extract_wavelengths(str(path), scans.dad, settings, [4,5,6], '#000000', cache=cache)
    assert cache.hits == 5 and cache.misses == 5
    with pytest.raises(Cancelled):
        extract_wavelengths(str(path), scans.dad, settings, [4,5,6], '#000000', is_cancelled=lambda: True)
    assert hashlib.sha256(path.read_bytes()).digest() == before


@pytest.mark.parametrize('band', [WavelengthBand(249, 4), WavelengthBand(260, 4), WavelengthBand(254.5, 0)])
def test_partial_or_empty_bands_are_not_silently_changed(run, band):
    path, _ = run
    with pytest.raises(ValueError, match='outside|No acquired'):
        extract_wavelengths(str(path), load_ms1(path).dad, [WavelengthExtraction(band)], [1], '#000000')


def test_empty_dad_spectrum_does_not_become_a_zero_twc(run, monkeypatch):
    path, _ = run
    monkeypatch.setattr(SpectrumReader, 'read', lambda *args: (np.array([]), np.array([])))
    with pytest.raises(ValueError, match='Empty DAD spectrum'):
        extract_wavelengths(str(path), load_ms1(path).dad, [WavelengthExtraction()], [1], '#000000')


def test_unknown_optical_units_rejected_only_when_optical_arrays_requested(run):
    path, _ = run
    path.write_bytes(path.read_bytes().replace(b'unitAccession="UO:0000269" unitName="absorbance unit"',
                                             b'unitAccession="UO:0000000" unitName="unqualified"'))
    scans = load_ms1(path)
    assert len(scans) == 16
    with SpectrumReader(path) as reader:
        assert reader.read(0)[0].size
        with pytest.raises(ValueError, match='absorbance unit'):
            reader.read(int(scans.dad.spectrum_index[0]))


@pytest.fixture
def viewer(qapp, tmp_path):
    paths = [write_mzml(acquisition(tmp_path, f'optical-{i}', offset=i*.01), tmp_path/f'optical-{i}.mzML') for i in range(2)]
    window = MainWindow()
    window.resize(1512, 930)
    window.show()
    assert not window.load_paths_sync([str(p) for p in paths])
    yield window, list(window.session.files.values())
    close(window)


def test_opt_in_dock_native_walk_sync_independent_visibility_and_colors(viewer):
    w, files = viewer
    assert w.dad_dock.isHidden() and not w.dad_file_ids
    assert all([t.kind for t in f.traces] == ['tic'] for f in files)
    settings = [WavelengthExtraction(), WavelengthExtraction(WavelengthBand(254, 4), WavelengthBand(258, 2))]
    w.extract_wavelength_sync(settings, [f.file_id for f in files])
    settled(w)
    assert w.dad_dock.isVisible() and len(w.dad_spectrum._frames) == 2
    trace = files[0].traces[1]
    w.inspect_rt(trace.trace_id, .54)
    settled(w)
    assert w._selected_rt == .55 and w.dad_spectrum._frames[0].rt == .55
    assert w.spectrum._frames[0].rt == pytest.approx(.5)
    w.walk_by(1)
    settled(w)
    assert w.dad_spectrum._frames[0].rt == .8
    w.set_trace_visibility({trace.trace_id})
    assert len(w.dad_spectrum._frames) == 2
    w.toggle_dad_spectrum(files[1].file_id)
    settled(w)
    assert len(w.dad_spectrum._frames) == 1 and files[1].traces[1].visible
    w.set_file_color(files[0].file_id, '#ff0000')
    assert w.dad_spectrum._frames[0].color == '#ff0000'
    assert w.spectrum._frames[0].color == '#ff0000'
    w.remove_traces({t.trace_id for t in files[0].traces if t.optical})
    assert len(w.dad_spectrum._frames) == 1
    w.close_file(files[0].file_id)
    settled(w)
    assert not w.dad_spectrum._frames


def test_mixed_overlay_native_units_match_y_hit_testing_and_svg(viewer):
    w, files = viewer
    w.extract_wavelength_sync([WavelengthExtraction()], [f.file_id for f in files])
    w.match_action.setChecked(True)
    w.overlay_action.setChecked(True)
    QTest.qWait(30)
    p = w.plots
    assert p._right_view is not None
    ion, optical = files[0].traces
    assert p._trace_view(ion.trace_id) is not p._trace_view(optical.trace_id)
    assert p._right_view.viewRange()[1][0] < 0
    np.testing.assert_array_equal(p._curves[optical.trace_id].yData, optical.y)
    w.match_action.setChecked(True)
    w.overlay_action.setChecked(False)
    assert p._trace_view(ion.trace_id).viewRange()[1][1] > p._trace_view(optical.trace_id).viewRange()[1][1]*10
    w.overlay_action.setChecked(True)
    w.inspect_rt(optical.trace_id, .55)
    settled(w)
    drag(w, optical, .3, .8)
    settled(w)
    assert w._selected_rt == pytest.approx(.8, abs=.02)
    exports = [ExportTrace(f.name, t.label, t.color, t.rt, t.y, t.unit) for f,t in w.session.ordered_traces()]
    for mode in ('stacked', 'overlay'):
        options = ExportOptions(layout=mode, xmin=0, xmax=1.5, width_in=10)
        root = ET.fromstring(render_chromatograms(exports, options))
        texts = [n.text for n in root.iter() if n.tag.endswith('text')]
        assert 'Ion Counts' in texts and 'Absorbance (mAU)' in texts
        assert not any(n.tag.rsplit('}',1)[-1] in {'g','image','clipPath','mask'} for n in root.iter())
        bounds = plot_geometry(exports, options)[3]
        assert len(bounds) == 2
        if mode == 'overlay':
            assert bounds['counts'][0] == 0 and bounds['mAU'][0] < 0


def test_floating_dad_spectrum_full_view_and_both_spectrum_exports(viewer):
    w, files = viewer
    w.extract_wavelength_sync([WavelengthExtraction()], [files[0].file_id])
    w.inspect_rt(files[0].traces[1].trace_id, .55)
    settled(w)
    w.dad_dock.setFloating(True)
    w.dad_dock.show()
    w.dad_dock.activateWindow()
    w.dad_spectrum.view.setFocus()
    w.dad_spectrum.plot.setXRange(253, 255, padding=0)
    QTest.qWait(20)
    QTest.keyClick(w.dad_spectrum.view, Qt.Key.Key_F)
    QTest.qWait(20)
    assert w.dad_spectrum.plot.viewRange()[0][0] < 250
    for optical in (True, False):
        dialog = w.spectrum_export_dialog(optical)
        try:
            assert dialog.svg_bytes and dialog.save_button.isEnabled(), dialog.error.text()
            assert ('Wavelength (nm)' if optical else 'm/z').encode() in dialog.svg_bytes
            assert ('Absorbance (mAU)' if optical else 'Ion Counts').encode() in dialog.svg_bytes
            dialog.layout_choice.setCurrentText('Overlay')
            dialog.update_preview()
            assert dialog.svg_bytes
        finally:
            dialog.deleteLater()


def test_no_dad_notification_on_load_only_on_extraction(qapp, tmp_path, monkeypatch):
    path = write_mzml(acquisition(tmp_path, dad=False), tmp_path/'ms-only.mzML')
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: messages.append(args[2]))
    w = MainWindow()
    try:
        assert not w.load_paths_sync([str(path)]) and not messages
        w.extract_wavelength_request([WavelengthExtraction()], list(w.session.files))
        assert len(messages) == 1 and 'No usable DAD' in messages[0]
        assert w.dad_dock.isHidden() and not w._busy
    finally:
        close(w)


@pytest.mark.parametrize('mode', ['stacked', 'overlay'])
def test_signed_chromatogram_paints_below_curve_after_zoom(qapp, mode):
    from PySide6.QtCore import QPointF
    from kmzviewer.gui.plots import ChromatogramPanel
    panel = ChromatogramPanel()
    x, y = np.array([0., 1., 2., 3.]), np.array([-8., -4., -4., -8.])
    original = y.copy()
    try:
        panel.resize(800, 400)
        panel.view.setBackground('white')
        panel.set_mode(mode)
        panel.set_traces([(1, 'Signed EWC', '#0072b2', x, y)], units={1: 'mAU'})
        panel.show()
        panel.autoscale_y = False
        panel._plots()[0].showGrid(x=False, y=False)
        # The trace minimum lies outside this view; shading still goes down.
        panel.set_x_range(1.1, 1.9)
        view = panel._trace_view(1)
        view.setYRange(-7, -1, padding=0)
        QTest.qWait(50)
        screenshot = panel.view.viewport().grab().toImage()
        def red(value):
            point = panel.view.mapFromScene(view.mapViewToScene(QPointF(1.5, value)))
            assert screenshot.rect().contains(point)
            return screenshot.pixelColor(point).red()
        assert red(-2) >= 250  # Above the negative plateau is unshaded.
        assert red(-6) < 245   # Below it has the trace's faint fill.
        np.testing.assert_array_equal(y, original)
    finally:
        panel.close()
        panel.deleteLater()
