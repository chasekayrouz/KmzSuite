"""Independent startup-review regressions for deferred heavy imports.

These tests preserve pre-deferral patch points and require newly deferred GUI
imports to report errors through the visible UI.
"""
import builtins
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.mzML"
    write_mzml(path, [dict(mz=[100., 200.], intensity=[4., 8.], rt=.1)])
    return path


def reject(*args, **kwargs):
    raise RuntimeError("review sentinel")


def test_viewer_reader_module_override_is_honored(sample, monkeypatch):
    from kmzviewer import spectra
    monkeypatch.setattr(spectra, "_UnitMzML", reject)
    with spectra.SpectrumReader(sample) as reader:
        reader._direct = False
        with pytest.raises(RuntimeError, match="review sentinel"):
            reader.read(0)


def test_viewer_vocabulary_module_override_is_honored(monkeypatch):
    from kmzcore import psi_ms
    from kmzviewer import spectra
    monkeypatch.setattr(psi_ms, "_vocabulary", None)
    monkeypatch.setattr(psi_ms, "load", reject)
    with pytest.raises(RuntimeError, match="review sentinel"):
        spectra._local_vocabulary()


def test_comet_reader_module_override_is_honored(sample, monkeypatch):
    from kmzcomet import reader
    monkeypatch.setattr(reader, "UnitMzML", reject)
    with pytest.raises(RuntimeError, match="review sentinel"):
        list(reader.iter_scans(sample))


def test_comet_first_use_returns_one_reader_class(monkeypatch):
    from kmzcomet import reader
    if not hasattr(reader, "unit_reader"):
        assert reader.UnitMzML is reader.UnitMzML
        return
    reader.unit_reader.cache_clear()
    original = builtins.__import__
    barrier = threading.Barrier(8)
    entered, release = threading.Event(), threading.Event()
    all_entered = threading.Event()
    callers = []

    def simultaneous(name, *args, **kwargs):
        if name == "pyteomics":
            callers.append(threading.get_ident())
            if len(callers) == 8:
                all_entered.set()
            entered.set()
            assert release.wait(timeout=5)
        return original(name, *args, **kwargs)

    def first_use(_):
        barrier.wait(timeout=5)
        return reader.unit_reader()

    monkeypatch.setattr(builtins, "__import__", simultaneous)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(first_use, i) for i in range(8)]
        try:
            assert entered.wait(timeout=5)
            # A serialized implementation has only one importer and is also valid.
            all_entered.wait(timeout=.2)
        finally:
            release.set()
        classes = [future.result(timeout=5) for future in futures]
    assert all(cls is classes[0] for cls in classes)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_deferred_svg_import_failure_is_visible(qapp, monkeypatch):
    from kmzviewer.gui.svg_preview import ExportDialog
    from kmzviewer.export_svg import ExportTrace
    trace = ExportTrace("sample", "TIC", "#0072b2", np.array([0., 1.]), np.array([4., 8.]))
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("review: Matplotlib unavailable")
        return original(name, *args, **kwargs)

    dialog = ExportDialog([(1, "sample", True)], (0., 1.), trace_data={1: trace})
    try:
        monkeypatch.setattr(builtins, "__import__", blocked)
        dialog.update_preview()
        assert dialog.error.text() == "review: Matplotlib unavailable"
        assert not dialog.save_button.isEnabled()
    finally:
        dialog._timer.stop()
        dialog.close()


def test_deferred_eic_import_failure_is_visible(qapp, monkeypatch):
    from kmzcomet_app.app import MainWindow
    original = builtins.__import__
    errors = []

    def blocked(name, *args, **kwargs):
        if name == "chromatograms" and kwargs.get("level", args[3] if len(args) > 3 else 0):
            raise ImportError("review: EIC renderer unavailable")
        return original(name, *args, **kwargs)

    window = MainWindow()
    window.result = SimpleNamespace(samples=[])
    monkeypatch.setattr(window, "_failed", errors.append)
    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        window._eics_ready(SimpleNamespace(entries=[], reused=0, files_read=0))
        assert errors == ["review: EIC renderer unavailable"]
        assert window.tabs.count() == 1
    finally:
        window.result = None
        window.close()
