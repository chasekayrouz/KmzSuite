"""GNPS scientific boundaries, privacy, offline snapshots and real Qt lifecycle."""
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from kmzcore.cancel import Cancelled
from kmzviewer.comparison import SpectrumFrame
from kmzviewer.gnps import (
    SearchSettings, load_analysis, matching_scans, new_analysis,
    prepare_query, save_analysis, validate_analysis,
)
from kmzviewer.gnps_client import GnpsClient, ServiceError, parse_hits, request_payload
from kmzviewer.gui.gnps import GnpsWindow
from kmzviewer.gui.app import MainWindow
from Tests.Viewer.mzml_factory import write_mzml
from Tests.Viewer.test_ms1_comparison import wait_for


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def scan(**overrides):
    values = dict(mz=np.array([100., 150., 200.]), intensity=np.array([5., 10., 20.]),
                  precursor_mz=300., charge=None, polarity=1, representation="centroid",
                  precursor_source="selected ion", file_name="private name.mzML", scan_id="scan=12",
                  spectrum_index=12, rt_min=2., collision_energy=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def row(**overrides):
    values = {"USI": "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001647", "Cosine": .95,
              "Matching Peaks": 8, "Delta Mass": 0., "CompoundName": "Synthetic candidate",
              "GNPSLibraryAccession": "CCMSLIB00000001647", "Dataset": "GNPS Library", "Adduct": "M+H", "Charge": 1}
    values.update(overrides)
    return values


def test_profile_query_uses_one_native_apex_per_plateau_without_mutation():
    original = scan(mz=np.arange(100., 108.), intensity=np.array([4., 1., 9., 9., 9., 1., 0., 3.]), representation="profile")
    before = original.intensity.copy()
    query = prepare_query(original)
    assert query["peaks"] == [[100., 4.], [103., 9.], [107., 3.]]
    assert "not centroid" in query["preparation"]
    assert np.array_equal(original.intensity, before)
    assert query["charge"] is None and query["collision_energy"] is None


@pytest.mark.parametrize("overrides", [
    dict(representation="unknown"), dict(precursor_mz=None), dict(precursor_mz=float("nan")),
    dict(intensity=np.array([1.])), dict(mz=np.array([200., 100., 150.]), representation="profile"),
    dict(intensity=np.array([1., float("nan"), 2.])), dict(intensity=np.zeros(3)),
])
def test_invalid_queries_are_not_silently_repaired(overrides):
    with pytest.raises(ValueError):
        prepare_query(scan(**overrides))


def test_centroid_sort_reorders_only_query_copy():
    original = scan(mz=np.array([200., 100., 150.]))
    assert prepare_query(original)["peaks"] == [[100., 10.], [150., 20.], [200., 5.]]
    assert original.mz.tolist() == [200., 100., 150.]


def test_selected_ion_overrides_isolation_and_filters_before_decode(tmp_path, monkeypatch):
    path = tmp_path / "scans.mzML"
    mass = 300.
    specs = [dict(mz=[100, 150], intensity=[10, 20], rt=rt, precursor_mz=mz, ms_level=2,
                  polarity=pol, centroid=True) for rt, mz, pol in
             [(20, mass, 1), (2.1, mass, -1), (2.2, mass, 1), (2.3, mass + mass * 20e-6, 1),
              (2.0, mass + mass * 20.001e-6, 1), (1., 400, 1)]]
    write_mzml(path, specs)
    text = path.read_text().replace('<precursor><selectedIonList', '<precursor><isolationWindow><cvParam accession="MS:1000827" value="299"/></isolationWindow><selectedIonList')
    path.write_text(text)
    import kmz2viewer.index as index
    decoded = []
    decode = index._decode_arrays
    def track(block):
        decoded.append(block)
        return decode(block)
    monkeypatch.setattr(index, "_decode_arrays", track)
    matches = matching_scans(path, mass, 2., 1)
    assert [s.rt_min for s in matches] == [2.2, 2.3, 20.]
    assert len(decoded) == 3
    assert all(s.precursor_source == "selected ion" and s.polarity == 1 and s.representation == "centroid" for s in matches)
    with pytest.raises(Cancelled):
        matching_scans(path, mass, 2., 1, cancel=lambda: True)


def test_payload_has_no_source_metadata_and_keeps_native_query():
    query = prepare_query(scan(), target_mz=300., target_rt=1.9)
    query["path"] = "/private/folder/secret.mzML"
    payload = request_payload(query, SearchSettings(), "analog")
    encoded = json.dumps(payload)
    assert not any(token in encoded for token in ("private", "secret", "scan=12", "rt_min", "source_name", "target_rt"))
    spectrum = json.loads(payload["query_spectrum"])
    assert set(spectrum) == {"peaks", "precursor_mz", "precursor_charge"}
    assert spectrum["precursor_charge"] == 0
    assert spectrum["peaks"][-1][1] == 100
    assert query["peaks"][-1][1] == 20
    assert payload["analog"] == "Yes" and payload["cache"] == "No"


def test_match_evidence_thresholds_analog_separation_and_malformed_response():
    data = {"results": [row(), row(USI="mzspec:x:y:scan:1", Cosine=.6),
                        row(USI="mzspec:x:y:scan:2", **{"Matching Peaks": 5}),
                        row(USI="mzspec:x:y:scan:3", **{"Delta Mass": 14.})]}
    assert len(parse_hits(data, SearchSettings(), "standard")) == 2
    assert [h["delta_mass"] for h in parse_hits(data, SearchSettings(), "analog")] == [14.]
    for bad in ({}, {"results": [row(Cosine=float("nan"))]}, {"results": [row(**{"Matching Peaks": 1.5})]}):
        with pytest.raises(ServiceError):
            parse_hits(bad, SearchSettings(), "standard")


def test_async_api_running_status_and_failure_are_not_empty_hits(monkeypatch):
    replies = iter([{"id": "test-id"}, {"status": "RUNNING"}, {"results": [row()]}])
    calls = []
    def transport(url, **kwargs):
        calls.append((url, kwargs))
        return json.dumps(next(replies)).encode()
    monkeypatch.setattr("kmzviewer.gnps_client.time.sleep", lambda _: None)
    client = GnpsClient(transport)
    result = client.search(prepare_query(scan()), SearchSettings(), "standard")
    assert len(result["hits"]) == 1 and result["database"] == "gnpslibrary"
    assert "payload" in calls[0][1] and calls[0][0].endswith("/search")
    replies = iter([{"id": "test-id"}, {"status": "FAILURE"}])
    with pytest.raises(ServiceError, match="complete"):
        client.search(prepare_query(scan()), SearchSettings(), "standard")


def test_snapshots_reopen_without_sources_and_atomic_failure_preserves_old(tmp_path, monkeypatch):
    analysis = new_analysis(prepare_query(scan()), SearchSettings())
    hit = parse_hits({"results": [row()]}, SearchSettings(), "standard")[0]
    analysis["searches"]["standard"] = dict(hits=[hit], error="")
    analysis["references"][hit["usi"]] = dict(peaks=[[100., 1.]], structure_png="")
    path = tmp_path / "result.kmzgnps"
    save_analysis(path, analysis)
    assert load_analysis(path) == analysis
    original = path.read_bytes()
    monkeypatch.setattr("kmzviewer.gnps.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError):
        save_analysis(path, analysis)
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]
    for broken in (dict(analysis, schema="future"), dict(analysis, query={}), dict(analysis, settings={"min_cosine": float("nan")})):
        with pytest.raises(ValueError):
            validate_analysis(broken)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.block = False
        self.started = Event()
        self.fail_analog = False

    def search(self, query, settings, mode, *, cancel, progress):
        self.calls.append(mode)
        self.started.set()
        while self.block:
            if cancel():
                raise Cancelled()
            time.sleep(.005)
        if mode == "analog" and self.fail_analog:
            raise ServiceError("Service temporarily unavailable")
        data = {"results": [row(**{"Delta Mass": 14. if mode == "analog" else 0.})]}
        return dict(hits=parse_hits(data, settings, mode), error="", created="2026-11-13T00:00:00+00:00")

    def reference(self, hit, *, cancel):
        self.calls.append("reference")
        return dict(peaks=[[100., 10.], [150., 4.]], precursor_mz=300., structure_png="", metadata={}, warnings=[])


def ready_window(client=None):
    window = GnpsWindow(client=client or FakeClient())
    window.show()
    window._loaded_scans([scan(), scan(rt_min=2.3, scan_id="scan=13")])
    return window


def shutdown(window):
    window.close()
    wait_for(lambda: window.runtime.closed)


def test_gui_search_both_auto_reference_and_saved_offline_open(qapp, tmp_path):
    client = FakeClient()
    window = ready_window(client)
    try:
        assert not client.calls
        window.search_button.click()
        wait_for(lambda: window.runtime.job is None and "reference" in client.calls)
        assert client.calls == ["standard", "analog", "reference"]
        assert window._tables["standard"].rowCount() == window._tables["analog"].rowCount() == 1
        window.dataset_button.click()
        wait_for(lambda: window.runtime.job is None)
        assert "datasets" in client.calls and window._tables["datasets"].rowCount() == 1
        path = tmp_path / "review.kmzgnps"
        save_analysis(path, window.analysis)
        other_client = FakeClient()
        reopened = GnpsWindow(client=other_client)
        try:
            reopened.open_analysis(path)
            QApplication.processEvents()
            assert other_client.calls == []
            assert reopened.query == window.query
            assert len(reopened.mirror.listDataItems()) == 2
            assert reopened.save_button.isEnabled()
        finally:
            shutdown(reopened)
        window.fields["min_cosine"].setValue(.9)
        assert window.analysis is None and window._tables["standard"].rowCount() == 0
        assert not window.save_button.isEnabled()
    finally:
        shutdown(window)


def test_gui_partial_failure_and_cancelled_stale_results(qapp):
    client = FakeClient()
    client.fail_analog = True
    window = ready_window(client)
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None and "reference" in client.calls)
        assert window.analysis["searches"]["standard"]["hits"]
        assert "failed" in window._summaries["analog"].text()
        client.block = True
        client.started.clear()
        window.search()
        wait_for(client.started.is_set)
        assert not window.scan_picker.isEnabled()
        window.cancel_button.click()
        wait_for(lambda: window.runtime.job is None)
        assert not window.analysis["searches"]
        assert window._tables["standard"].rowCount() == 0
    finally:
        shutdown(window)


def test_peak_menu_captures_source_mass_rt_and_polarity(qapp):
    window = MainWindow()
    window.show()
    captured = []
    window.spectrum.gnps_requested.connect(lambda *values: captured.append(values))
    try:
        frame = SpectrumFrame(1, "Source", "#008080", "Source", np.array([300.123456789]), np.array([50.]), True, rt=2.5, polarity=-1)
        window.spectrum.show_frames([frame], "Scan")
        QApplication.processEvents()
        item = window.spectrum._label_items[0]
        position = item.textItem.mapToScene(item.textItem.boundingRect().center())
        menu = window.spectrum._label_context_menu(position)
        frame.rt, frame.polarity = 20., 1
        action = next(a for a in menu.actions() if a.text() == "Spectral analysis…")
        action.trigger()
        assert captured == [(1, 300.123456789, 2.5, -1)]
        menu.close()
    finally:
        shutdown(window)


def test_parent_close_cancels_and_waits_for_gnps_worker(qapp):
    parent = MainWindow()
    parent.show()
    window = parent._new_gnps_window()
    client = FakeClient()
    client.block = True
    window.client = client
    window._loaded_scans([scan()])
    window.search()
    wait_for(client.started.is_set)
    parent.close()
    wait_for(lambda: parent.runtime.closed)
    assert not parent._gnps_windows
