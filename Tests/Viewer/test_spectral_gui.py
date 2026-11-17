"""Offline exploration, explicit persistence and worker lifetime integration."""
from copy import deepcopy
from dataclasses import asdict
import json
import sys
from threading import Event
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from kmzcore.cancel import Cancelled
from kmzviewer.gnps import SearchSettings, new_analysis, prepare_query, save_analysis
from kmzviewer.gui.app import MainWindow
from kmzviewer.gui.spectral import SpectralWindow
from kmzviewer.gui.spectral_smoke import fixture_motifs, fixture_pack
from kmzviewer.spectral_analysis import (
    load_spectral_analysis, new_spectral_analysis, save_spectral_analysis, validate_spectral_analysis,
)
from kmzviewer.spectral_matching import SpectralSettings, search_reference_pack
from Tests.Viewer.test_ms1_comparison import wait_for
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def scan(**changes):
    fields = dict(mz=np.array([100., 150., 180., 220.]), intensity=np.array([10., 50., 30., 20.]),
        precursor_mz=300., charge=1, polarity=1, representation="centroid", precursor_source="selected ion",
        file_name="synthetic.mzML", scan_id="scan=1", spectrum_index=1, rt_min=2., collision_energy=None)
    fields.update(changes)
    return SimpleNamespace(**fields)


def analysis():
    query = prepare_query(scan())
    settings = SpectralSettings()
    return new_spectral_analysis(query, settings, search_reference_pack(query, settings, fixture_pack(query)),
                                 fixture_motifs(query, settings))


class Backend:
    def __init__(self):
        self.calls = []
        self.started = Event()
        self.block = False
        self.fail = False

    def load(self, path, **kwargs):
        self.calls.append("load")
        if self.fail:
            raise ValueError("Reference download unavailable")
        return fixture_pack(prepare_query(scan()))

    def search(self, query, settings, pack, *, cancel=None, progress=None):
        self.calls.append("search")
        self.started.set()
        while self.block:
            if cancel():
                raise Cancelled()
            time.sleep(.005)
        return search_reference_pack(query, settings, pack, cancel=cancel, progress=progress)


def shutdown(window):
    window.close()
    wait_for(lambda: window.runtime.closed)


def window_for(tmp_path, backend=None):
    backend = backend or Backend()
    window = SpectralWindow(cache_dir=tmp_path, pack_loader=backend.load, searcher=backend.search,
                            motif_searcher=fixture_motifs)
    window.show()
    window._loaded_scans([scan(), scan(scan_id="scan=2", rt_min=2.2)])
    return window, backend


def test_local_search_displays_evidence_then_reopens_offline(qapp, tmp_path, monkeypatch):
    requests = []
    def forbidden(*args, **kwargs):
        requests.append(args)
        raise AssertionError("No automatic network request is allowed")
    monkeypatch.setattr("kmzviewer.gui.spectral.qt_request", forbidden)
    window, backend = window_for(tmp_path)
    before = deepcopy(window.query)
    try:
        assert backend.calls == [] and requests == []
        window.search_button.click()
        wait_for(lambda: window.runtime.job is None)
        assert backend.calls == ["load", "search"]
        assert window.query == before
        assert window.matches.rowCount() == 2 and window.motifs.rowCount() == 1
        assert window.pairs.rowCount() == 4
        assert "100.0%" in window.evidence.text()
        assert window.save_button.isEnabled()
        path = tmp_path / "analysis.kmzspectra"
        save_spectral_analysis(path, window.analysis)
        reopened = SpectralWindow(cache_dir=tmp_path, pack_loader=forbidden, searcher=forbidden,
                                  motif_searcher=forbidden)
        try:
            reopened.open_analysis(path)
            QApplication.processEvents()
            assert reopened.query == before
            assert reopened.matches.rowCount() == 2
            reopened.tabs.setCurrentIndex(1)
            assert reopened.pairs.rowCount() == 3
            assert reopened.pairs.item(0, 1).text() == "100.00000"
            assert requests == []
        finally:
            shutdown(reopened)
    finally:
        shutdown(window)


def test_ranking_switch_uses_correct_direct_and_shifted_alignment(qapp, tmp_path):
    window, _ = window_for(tmp_path)
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None)
        window.ranking.setCurrentIndex(window.ranking.findData("neutral_loss"))
        shifted_row = next(i for i, h in enumerate(window._rows) if h["accession"] == "SYNTHETIC2")
        window.matches.selectRow(shifted_row)
        assert window.pairs.rowCount() == 4
        assert all(window.pairs.item(i, 2).text() == "precursor difference" for i in range(4))
        window.ranking.setCurrentIndex(window.ranking.findData("direct"))
        window.matches.selectRow(0)
        assert all(window.pairs.item(i, 2).text() == "fragment" for i in range(4))
    finally:
        shutdown(window)


def test_scan_and_settings_changes_clear_stale_results(qapp, tmp_path):
    window, backend = window_for(tmp_path)
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None)
        window.fragment.setValue(.03)
        assert window.analysis is None and window.matches.rowCount() == 0
        assert not window.save_button.isEnabled()
        window.search()
        wait_for(lambda: window.runtime.job is None)
        window.scan_picker.setCurrentIndex(1)
        assert window.query["scan_id"] == "scan=2"
        assert window.analysis is None
        # A reference pack survives repeated searches; only its preparation index
        # is rebuilt for changed tolerances, avoiding repeated download/decoding.
        assert backend.calls == ["load", "search", "search"]
    finally:
        shutdown(window)


def test_no_ms2_and_invalid_required_mass_do_not_start_work(qapp, tmp_path):
    window, backend = window_for(tmp_path)
    try:
        window._loaded_scans([])
        assert not window.search_button.isEnabled()
        assert window.query is None
        assert "No matching acquired MS2" in window.scan_details.text()
        window._loaded_scans([scan()])
        window.fragments.setText("not-a-mass")
        window.search()
        assert backend.calls == [] and window.runtime.job is None
        assert "numbers separated" in window.statusBar().currentMessage()
    finally:
        shutdown(window)


def test_reference_failure_retains_independent_motif_evidence(qapp, tmp_path):
    backend = Backend()
    backend.fail = True
    window, _ = window_for(tmp_path, backend)
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None)
        assert "unavailable" in window.summary.text()
        assert window.motifs.rowCount() == 1
        assert window.tabs.currentIndex() == 1
        assert window.save_button.isEnabled()
    finally:
        shutdown(window)


def test_cancelled_search_does_not_apply_late_results(qapp, tmp_path):
    backend = Backend()
    backend.block = True
    window, _ = window_for(tmp_path, backend)
    try:
        window.search()
        wait_for(backend.started.is_set)
        assert not window.scan_picker.isEnabled()
        window.cancel_button.click()
        wait_for(lambda: window.runtime.job is None)
        assert window.analysis is None and window.matches.rowCount() == 0
        assert window.search_button.isEnabled()
        assert not window.save_button.isEnabled()
    finally:
        shutdown(window)


def test_parent_close_waits_for_spectral_worker(qapp, tmp_path):
    parent = MainWindow()
    parent.show()
    window = parent._new_spectral_window()
    backend = Backend()
    backend.block = True
    window.cache_dir = tmp_path
    window.pack_loader, window.searcher, window.motif_searcher = backend.load, backend.search, fixture_motifs
    window._loaded_scans([scan()])
    window.search()
    wait_for(backend.started.is_set)
    parent.close()
    wait_for(lambda: parent.runtime.closed)
    assert not parent._gnps_windows


def test_packaged_smoke_step_exercises_source_read_and_saved_evidence(qapp, tmp_path):
    from kmzviewer.gui.spectral_smoke import step
    path = tmp_path / "synthetic.mzML"
    write_mzml(path, [dict(mz=[100., 150., 180., 220.], intensity=[10., 50., 30., 20.],
        rt=1., precursor_mz=300., ms_level=2, polarity=1, centroid=True)])
    parent = MainWindow()
    parent.show()
    state = {}
    try:
        wait_for(lambda: step(parent, path, tmp_path, state), timeout=12)
        assert (tmp_path / "synthetic.kmzspectra").is_file()
        assert (tmp_path / "spectral-synthetic.png").is_file()
        assert not parent._gnps_windows
    finally:
        shutdown(parent)


def test_legacy_gnps_import_is_query_only_and_does_not_submit(qapp, tmp_path):
    query = prepare_query(scan())
    path = tmp_path / "legacy.kmzgnps"
    save_analysis(path, new_analysis(query, SearchSettings()))
    window, backend = window_for(tmp_path)
    try:
        window.open_analysis(path)
        assert window.query == query and window.analysis is None
        assert backend.calls == []
        assert "Imported the saved GNPS query locally" in window.statusBar().currentMessage()
        window.open_online()
        online = window._online_windows[0]
        assert online.query == query
        assert online.runtime.job is None and online.analysis is None
    finally:
        shutdown(window)


def learned_fixture():
    hits, references = [], {}
    for index, (name, score, peaks) in enumerate([
        ("Synthetic learned neighbor", .91, [[41., 10.], [61., 20.], [81., 30.]]),
        ("Synthetic direct neighbor", .62, [[100., 10.], [150., 50.], [180., 30.], [220., 20.]])]):
        accession = f"MassSpecGymID{index + 1:07d}"
        metadata = dict(name=name, accession=accession, precursor_mz=300., scores={"dreams": score}, score=score,
            library="Synthetic learned qualification", smiles="", adduct="", provenance="Artificial reference spectrum")
        hits.append(metadata)
        references[accession] = dict(peaks=peaks, precursor_mz=300.)
    return dict(hits=hits, references=references, provenance=dict(description="Synthetic learned-search qualification"),
                preparation={}, warnings=[])


def install_learned_stub(monkeypatch, search):
    # Keep these UI tests independent of service availability and model/index
    # packages. The real service's payload boundary has separate unit coverage.
    module = ModuleType("kmzviewer.dreams_client")
    module.search_dreams = search
    monkeypatch.setitem(sys.modules, "kmzviewer.dreams_client", module)


def test_learned_search_requires_checkbox_and_search_and_keeps_its_own_ranking(qapp, tmp_path, monkeypatch):
    calls = []
    def learned(query, cache, *, cancel, progress):
        calls.append(deepcopy(query))
        progress("Synthetic learned search")
        return learned_fixture()
    install_learned_stub(monkeypatch, learned)
    window, _ = window_for(tmp_path)
    try:
        assert not window.learned_enabled.isChecked()
        window.ranking.setCurrentIndex(window.ranking.findData("dreams"))
        assert calls == []
        window.search()
        wait_for(lambda: window.runtime.job is None)
        assert calls == [] and not window.analysis["learned"]
        assert "Enable DreaMS" in window.summary.text()
        window.learned_enabled.setChecked(True)
        assert calls == [] and window.runtime.job is None
        assert "sends selected peaks" in window.privacy.text()
        window.search_button.click()
        wait_for(lambda: window.runtime.job is None)
        assert len(calls) == 1
        assert window.matches.rowCount() == 2
        assert "learned embedding similarity" in window.summary.text()
        assert "entropy similarity" not in window.summary.text()
        # The independently displayed peak alignment must not rerank learned hits.
        assert window.matches.item(0, 0).text() == "Synthetic learned neighbor"
        assert window.matches.item(0, 1).text() == "0.910"
        assert window.matches.item(0, 2).text() == "0"
        window.matches.selectRow(1)
        assert window.matches.item(1, 1).text() == "0.620"
        assert window.pairs.rowCount() == 4
        assert "DreaMS 0.620" in window.evidence.text()
    finally:
        shutdown(window)


def test_saved_learned_neighbors_reopen_without_service_or_reference_pack(qapp, tmp_path, monkeypatch):
    calls = []
    def learned(query, cache, *, cancel, progress):
        calls.append("learned")
        return learned_fixture()
    install_learned_stub(monkeypatch, learned)
    window, _ = window_for(tmp_path)
    try:
        window.learned_enabled.setChecked(True)
        window.search()
        wait_for(lambda: window.runtime.job is None)
        path = tmp_path / "learned.kmzspectra"
        save_spectral_analysis(path, window.analysis)
        def forbidden(*args, **kwargs):
            raise AssertionError("Reopening must not access any service or reference pack")
        install_learned_stub(monkeypatch, forbidden)
        reopened = SpectralWindow(cache_dir=tmp_path / "absent", pack_loader=forbidden,
                                  searcher=forbidden, motif_searcher=forbidden)
        try:
            reopened.open_analysis(path)
            assert reopened.learned_enabled.isChecked()
            reopened.ranking.setCurrentIndex(reopened.ranking.findData("dreams"))
            QApplication.processEvents()
            assert reopened.runtime.job is None
            assert reopened.matches.rowCount() == 2
            assert reopened.matches.item(0, 1).text() == "0.910"
            reopened.matches.selectRow(1)
            assert reopened.pairs.rowCount() == 4
            assert calls == ["learned"]
        finally:
            shutdown(reopened)
    finally:
        shutdown(window)


def test_learned_service_failure_preserves_local_results(qapp, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Learned service unavailable")
    install_learned_stub(monkeypatch, fail)
    window, _ = window_for(tmp_path)
    try:
        window.learned_enabled.setChecked(True)
        window.search()
        wait_for(lambda: window.runtime.job is None)
        assert window.matches.rowCount() == 2 and window.motifs.rowCount() == 1
        window.ranking.setCurrentIndex(window.ranking.findData("dreams"))
        assert window.matches.rowCount() == 0
        assert "no reference or motif selected" in window.legend.text()
        assert window.footer.text() == ""
        assert "Learned service unavailable" in window.summary.text()
        assert window.save_button.isEnabled()
    finally:
        shutdown(window)


def test_snapshot_roundtrip_atomic_failure_and_size_limit(tmp_path, monkeypatch):
    data = analysis()
    path = tmp_path / "analysis.kmzspectra"
    save_spectral_analysis(path, data)
    assert load_spectral_analysis(path) == json.loads(json.dumps(data))
    old = path.read_bytes()
    def fail(*_):
        raise OSError("disk failure")
    monkeypatch.setattr("kmzviewer.spectral_analysis.os.replace", fail)
    with pytest.raises(OSError):
        save_spectral_analysis(path, data)
    assert path.read_bytes() == old and list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr("kmzviewer.spectral_analysis.MAX_BYTES", 100)
    with pytest.raises(ValueError, match="snapshot limit"):
        load_spectral_analysis(path)


@pytest.mark.parametrize("mutation", [
    lambda d: d["library"]["methods"]["hybrid"][0].update(score="not a number"),
    lambda d: d["library"]["methods"]["hybrid"][0]["scores"].update(hybrid=2.),
    lambda d: d["library"]["methods"]["hybrid"][0]["alignment"]["pairs"][0].update(query_mz="bad"),
    lambda d: d["motifs"]["hits"][0].update(motif_coverage=-1.),
    lambda d: d["library"]["methods"]["hybrid"][0]["scores"].update(hybrid=None),
    lambda d: next(iter(d["library"]["references"].values())).update(comparison_peaks=[[100., "bad"]]),
])
def test_snapshot_rejects_malformed_numeric_evidence_before_display(mutation):
    data = analysis()
    mutation(data)
    with pytest.raises(ValueError):
        validate_spectral_analysis(data)


@pytest.mark.parametrize("kind, reference, expected", [("fragment", 100.01, "-0.01000"), ("loss", 199.98, "+0.02000")])
def test_motif_evidence_reports_signed_observed_minus_reference(qapp, tmp_path, kind, reference, expected):
    window, _ = window_for(tmp_path)
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None)
        pair = window.analysis["motifs"]["hits"][0]["matches"][0]
        pair.update(type=kind, query_mz=100., reference_mz=reference, error_da=abs(float(expected)))
        window.tabs.setCurrentIndex(1)
        assert window.pairs.item(0, 3).text() == expected
    finally:
        shutdown(window)


def test_dense_reference_preparation_runs_on_search_worker(qapp, tmp_path):
    """Ordered native peaks exceeded macOS's default QThread stack."""
    query = prepare_query(scan())
    dense = dict(query, peaks=np.column_stack((np.linspace(40., 240., 30000),
                                               np.linspace(1., 100., 30000))).tolist())
    pack = fixture_pack(dense)
    window = SpectralWindow(cache_dir=tmp_path, pack_loader=lambda *a, **k: pack,
                            motif_searcher=fixture_motifs)
    window.query = query
    try:
        window.search()
        wait_for(lambda: window.runtime.job is None, timeout=15.)
        assert window.analysis is not None
        assert not window.analysis['library'].get('error')
        assert len(window.analysis['library']['methods']) == 3
    finally:
        shutdown(window)
