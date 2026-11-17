"""Independent native-point checks for BPC and reversible m/z-filter views."""
import hashlib
import time
from threading import Event

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from kmzcore.cancel import Cancelled
from kmzcore.runtime.synthetic import write_fixture
from kmzviewer.extract import MassList, MzRange, extract_bpc, extract_eics, make_tic_trace
from kmzviewer.export_svg import ExportOptions
from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.dialogs import BpcDialog
from kmzviewer.model import Trace
from kmzviewer.mz_filter import FilterCache, extract_filtered_values, mz_range
from kmzviewer.mzml_index import index_ms1
from kmzviewer.spectra import ArrayCache
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "ions.mzML"
    write_mzml(path, [
        dict(mz=[300., 100., 200., 200., np.nan], intensity=[90., 2., 3., 4., 1000.], rt=1., tic=500., polarity=1),
        dict(mz=[100.], intensity=[9999.], rt=1.1, tic=9999., ms_level=2),
        dict(mz=[100., 200., 300.], intensity=[5., 6., np.inf], rt=1., tic=600., polarity=-1, centroid=True),
        dict(mz=[], intensity=[], rt=3., tic=0., polarity=1),
        dict(mz=[100., 200.], intensity=[-2., -3.], rt=4., tic=-5., polarity=0),
    ])
    return str(path), index_ms1(path)


@pytest.mark.parametrize("polarity,rows,expected", [
    ("all", [0, 1, 2, 3], [90., 6., 0., -2.]),
    ("positive", [0, 2], [90., 0.]), ("negative", [1], [6.]),
])
def test_bpc_uses_native_maxima_with_scan_identity(sample, polarity, rows, expected):
    path, scans = sample
    trace = extract_bpc(path, scans, polarity, 10, "#0072b2")
    np.testing.assert_array_equal(trace.y, expected)
    np.testing.assert_array_equal(trace.row, rows)
    np.testing.assert_array_equal(trace.rt, scans.rt_min[rows])
    assert trace.kind == "bpc" and trace.extraction_key == ("bpc", polarity)


def test_filter_intersects_exact_windows_preserving_overlap_and_native_arrays(sample):
    path, scans = sample
    tic = make_tic_trace(1, "#0072b2", scans)
    bpc = extract_bpc(path, scans, "positive", 2, "#0072b2")
    merged = extract_eics(path, scans, MassList((200., 100., 200.), 0.), "all", True, [3], ["#0072b2"])[0]
    outside = extract_eics(path, scans, MzRange(299., 301.), "negative", False, [4], ["#0072b2"])[0]
    optical = Trace(5, 0, "twc", "TWC", np.array([1.]), np.array([-4.]), np.array([0]), "optical", "#0072b2")
    traces = [tic, bpc, merged, outside, optical]
    before = [t.y.copy() for t in traces]
    filtered = extract_filtered_values(path, scans, traces, mz_range(100., 200.))
    np.testing.assert_array_equal(filtered[1], [9., 11., 0., -5.])
    np.testing.assert_array_equal(filtered[2], [4., 0.])
    np.testing.assert_array_equal(filtered[3], [16., 17., 0., -8.])
    np.testing.assert_array_equal(filtered[4], [0.])
    assert 5 not in filtered
    for trace, original in zip(traces, before):
        np.testing.assert_array_equal(trace.y, original)
    exact = extract_filtered_values(path, scans, [tic, merged], mz_range(200., 200.))
    np.testing.assert_array_equal(exact[1], [7., 6., 0., -3.])
    np.testing.assert_array_equal(exact[3], [14., 12., 0., -6.])


def test_filter_matches_float64_slices_and_merged_input_order(tmp_path):
    path = tmp_path / "roundoff.mzML"
    write_mzml(path, [dict(mz=[100., 200., 300., 400.], intensity=[1e30, 1e-10, -1e30, 2e-10], rt=1.)])
    scans = index_ms1(path)
    traces = extract_eics(str(path), scans, MassList((100., 300., 200., 400., 200.), 0.), "all", True, [1], ["#0072b2"])
    result = extract_filtered_values(str(path), scans, traces, mz_range(0, 1000))
    np.testing.assert_array_equal(result[1].view(np.uint64), traces[0].y.view(np.uint64))
    tiny = extract_filtered_values(str(path), scans, traces, mz_range(200, 200))
    assert tiny[1][0] == 2e-10


def test_filter_reads_once_per_scan_and_reuses_array_cache(sample, monkeypatch):
    import kmzviewer.mz_filter as module
    path, scans = sample
    traces = [make_tic_trace(1, "#0072b2", scans), extract_bpc(path, scans, "all", 2, "#0072b2")]
    original = module.SpectrumReader.read
    reads = []
    def read(reader, index):
        reads.append(index)
        return original(reader, index)
    monkeypatch.setattr(module.SpectrumReader, "read", read)
    cache = ArrayCache()
    extract_filtered_values(path, scans, traces, mz_range(100, 200), cache=cache)
    assert reads == scans.spectrum_index.tolist()
    extract_filtered_values(path, scans, traces, mz_range(200, 300), cache=cache)
    # The existing array cache deliberately does not retain zero-byte scans.
    assert reads == scans.spectrum_index.tolist() + [int(scans.spectrum_index[2])]


def test_disjoint_eic_needs_no_spectral_read_and_cancellation_is_atomic(sample, monkeypatch):
    import kmzviewer.mz_filter as module
    path, scans = sample
    traces = extract_eics(path, scans, MzRange(100, 200), "all", False, [1], ["#0072b2"])
    def fail(*_):
        raise AssertionError("No decode should be needed")
    monkeypatch.setattr(module.SpectrumReader, "read", fail)
    result = extract_filtered_values(path, scans, traces, mz_range(400, 500))
    np.testing.assert_array_equal(result[1], np.zeros(len(scans)))
    with pytest.raises(Cancelled):
        extract_filtered_values(path, scans, traces, mz_range(100, 200), is_cancelled=lambda: True)


@pytest.mark.parametrize("lower,upper", [(np.nan, 200), (100, np.inf), (-1, 2), (200, 100)])
def test_invalid_filter_bounds(lower, upper):
    with pytest.raises(ValueError):
        mz_range(lower, upper)


def test_completed_result_cache_bounds_eviction_and_trace_release():
    cache = FilterCache(budget_bytes=32, limit=2)
    bounds = mz_range(100, 200)
    a, b, c = (np.array([i, i], dtype=float) for i in range(3))
    cache.put(1, bounds, a)
    cache.put(2, bounds, b)
    assert cache.get(1, bounds) is a
    cache.put(3, bounds, c)
    assert cache.get(2, bounds) is None
    assert cache.nbytes == 32
    cache.drop_traces({1, 3})
    assert cache.nbytes == 0
    cache.put(4, bounds, np.zeros(100))
    assert cache.get(4, bounds) is None


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_idle(window, app):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if not window._busy and not window._filter_refresh_pending:
            return
        time.sleep(.005)
    raise AssertionError("Background work did not settle")


def test_gui_filtered_view_export_clear_cache_and_new_traces(qapp, tmp_path, monkeypatch):
    import kmzviewer.mz_filter as module
    path = tmp_path / "generic.mzML"
    write_fixture(path, include_dad=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    window = MainWindow()
    window.show()
    try:
        assert window.load_paths_sync([str(path)]) == []
        loaded = next(iter(window.session.files.values()))
        window.extract_bpc_request("positive", [loaded.file_id])
        wait_idle(window, qapp)
        assert loaded.traces[-1].y.tolist() == [5., 8.]
        originals = {t.trace_id: t.y for t in loaded.traces}
        original_curves = dict(window.plots._curves)
        original_panes = list(window.plots._plots())
        window.plots.set_x_range(1.1, 1.8)
        window.mz_filter.start.setText("2e2")
        window.mz_filter.stop.setText("200")
        assert window.mz_filter.clear_button.isEnabled()
        window.mz_filter.apply_button.click()
        wait_idle(window, qapp)
        assert [row[4].tolist() for row in window.plots._rows] == [[2., 3.], [2., 3.]]
        assert all("m/z 200–200" in row[1] for row in window.plots._rows)
        assert window.plots._curves == original_curves
        assert window.plots._plots() == original_panes
        assert window.plots.current_x_range() == pytest.approx((1.1, 1.8))
        for t in loaded.traces:
            assert t.y is originals[t.trace_id]
        from kmzviewer.dad import WavelengthExtraction
        window.extract_wavelength_sync([WavelengthExtraction()], [loaded.file_id])
        wait_idle(window, qapp)
        optical = loaded.traces[-1]
        assert window._display_trace(optical) is optical
        svg = tmp_path / "filtered.svg"
        window.export_trace_ids(list(originals), str(svg), ExportOptions(xmin=0, xmax=3))
        assert "m/z 200–200" in svg.read_text()
        window.export_trace_ids(list(originals), str(svg), ExportOptions(xmin=0, xmax=3, legend=False))
        assert "m/z 200–200" in svg.read_text()
        for mode in (True, False):
            window.overlay_action.setChecked(mode)
            qapp.processEvents()
            assert window.plots._rows[0][4].tolist() == [2., 3.]
        # A new EIC inherits the active filter, including a disjoint zero result.
        window.extract_sync("100", 0., "all", False, [loaded.file_id])
        wait_idle(window, qapp)
        eic = loaded.traces[-1]
        assert eic.y.tolist() == [5., 8.]
        assert window._display_trace(eic).y.tolist() == [0., 0.]
        window.mz_filter.clear_button.click()
        assert window._mz_range is None
        assert window.plots._rows[0][4] is originals[loaded.traces[0].trace_id]
        # Reapplying the completed range must not create a worker or decode data.
        def fail(*_):
            raise AssertionError("Completed filtered traces should be cached")
        monkeypatch.setattr(module.SpectrumReader, "read", fail)
        window.apply_mz_filter(200, 200)
        assert not window._busy
        assert "cached" in window.mz_filter.status.text()
        removed = eic.trace_id
        window.remove_trace(removed)
        assert window._filter_cache.get(removed, mz_range(200, 200)) is None
        window.close_file(loaded.file_id)
        assert not window._filter_values and window._filter_cache.nbytes == 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    finally:
        window.close()
        wait_idle(window, qapp)


def test_clear_during_background_decode_rejects_late_result(qapp, tmp_path, monkeypatch):
    import kmzviewer.mz_filter as module
    path = tmp_path / "generic.mzML"
    write_fixture(path)
    window = MainWindow()
    entered, release = Event(), Event()
    original = module.SpectrumReader.read
    def delayed(reader, index):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Test decode was not released")
        return original(reader, index)
    monkeypatch.setattr(module.SpectrumReader, "read", delayed)
    try:
        window.load_paths_sync([str(path)])
        window.apply_mz_filter(200, 200)
        deadline = time.monotonic() + 5
        while not entered.is_set() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.005)
        assert entered.is_set()
        window.clear_mz_filter()
        assert window.plots._rows[0][4].tolist() == [40., 50.]
        release.set()
        wait_idle(window, qapp)
        assert window._mz_range is None and not window._filter_values
    finally:
        release.set()
        window.close()
        wait_idle(window, qapp)


def test_bpc_dialog_file_scope_and_polarity(qapp):
    dialog = BpcDialog([(1, "A"), (2, "B")], selected_file_ids=[2])
    try:
        assert dialog.checked_file_ids() == [2]
        assert [dialog.polarity.itemData(i) for i in range(3)] == ["all", "positive", "negative"]
        dialog._check_files(Qt.CheckState.Unchecked)
        assert not dialog.ok_button.isEnabled()
    finally:
        dialog.close()


def test_failed_range_keeps_previous_view_and_invalid_input_does_not_apply(qapp, tmp_path, monkeypatch):
    import kmzviewer.mz_filter as module
    path = tmp_path / "generic.mzML"
    write_fixture(path)
    window = MainWindow()
    errors = []
    monkeypatch.setattr(window, "_on_failed", errors.append)
    try:
        window.load_paths_sync([str(path)])
        window.apply_mz_filter(200, 200)
        wait_idle(window, qapp)
        prior = window.plots._rows[0][4]
        window.mz_filter.start.setText("nan")
        window.mz_filter.apply_button.click()
        assert window._mz_range == mz_range(200, 200)
        assert "unchanged" in window.mz_filter.status.text()
        window._array_cache.drop_path(str(path))
        def fail(*_):
            raise ValueError("Synthetic decode failure")
        monkeypatch.setattr(module.SpectrumReader, "read", fail)
        window.apply_mz_filter(100, 100)
        assert window.plots._rows[0][4] is prior
        wait_idle(window, qapp)
        assert errors == ["Synthetic decode failure"]
        assert window._mz_range == mz_range(200, 200)
        assert window.plots._rows[0][4] is prior
        assert "not applied" in window.mz_filter.status.text()
    finally:
        window.close()
        wait_idle(window, qapp)
