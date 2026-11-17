"""Offline, synthetic GNPS coverage for the existing frozen-app smoke runner."""
from kmzviewer.gnps import save_analysis
from kmzviewer.gnps_client import parse_hits


class SyntheticClient:
    def __init__(self):
        self.calls = []

    def search(self, query, settings, mode, **kwargs):
        self.calls.append(mode)
        rows = [{"USI": "mzspec:synthetic:reference:scan:1", "Cosine": .9,
                 "Matching Peaks": 6, "Delta Mass": 14. if mode == "analog" else 0.,
                 "CompoundName": "Synthetic reference", "Dataset": "Synthetic dataset", "Adduct": "M+H"}]
        return dict(hits=parse_hits({"results": rows}, settings, mode), error="", created="2026-01-01T00:00:00Z")

    def reference(self, hit, **kwargs):
        self.calls.append("reference")
        return dict(peaks=[[100., 30.], [132.076, 100.], [180., 50.]], structure_png="", metadata={}, warnings=[])


def step(main, source, output, state):
    """Advance once per GUI timer tick, returning True only after safe close."""
    if "window" not in state:
        from PySide6.QtNetwork import QSslSocket
        assert QSslSocket.supportsSsl(), "Packaged GNPS transport needs a TLS backend"
        window = main._new_gnps_window()
        window.client = SyntheticClient()
        window.show()
        window.load_source(source, 300., 1., 1)
        state.update(window=window, phase=0)
        return False
    window = state["window"]
    if window.runtime.job is not None:
        return False
    if state["phase"] == 0:
        assert window.query and window.query["precursor_mz"] == 300.
        assert not window.client.calls
        window.search_button.click()
        state["phase"] = 1
    elif state["phase"] == 1:
        if "reference" not in window.client.calls:
            return False
        assert window.client.calls == ["standard", "analog", "reference"]
        assert all(window._tables[mode].rowCount() == 1 for mode in ("standard", "analog"))
        window.dataset_button.click()
        state["phase"] = 2
    elif state["phase"] == 2:
        assert window._tables["datasets"].rowCount() == 1
        path = output / "synthetic.kmzgnps"
        save_analysis(path, window.analysis)
        calls = list(window.client.calls)
        window.open_analysis(path)
        assert window.client.calls == calls and window.offline
        assert len(window.mirror.listDataItems()) == 2
        window.grab().save(str(output / "gnps-synthetic.png"))
        window.close()
        state["phase"] = 3
    else:
        return window not in main._gnps_windows
    return False


def run_live(app, main, output):
    """Explicit opt-in frozen HTTPS check. Never uses a local acquisition query."""
    import json
    from pathlib import Path
    import sys
    import time
    from PySide6.QtCore import QTimer
    from kmzviewer.gui.gnps import GnpsWorker

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    app.setQuitOnLastWindowClosed(False)
    window = main._new_gnps_window()
    window.show()
    state = {"phase": 0, "started": time.monotonic()}
    public = dict(usi="mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001647",
                  accession="CCMSLIB00000001647", name="Enterocin", adduct="M+Na")

    def loaded(reference):
        window.query = dict(precursor_mz=reference["precursor_mz"], charge=reference["charge"],
            polarity=1, peaks=reference["peaks"], preparation="Public GNPS reference peak list",
            source_name="Public GNPS reference", scan_id=public["accession"], rt_min=None,
            collision_energy=None, precursor_source="public spectrum resolver")
        window.source.setText("Live qualification · fixed public GNPS reference")
        window._show_query()
        state["loaded"] = True

    window.runtime.start(GnpsWorker(lambda cancel, progress: window.client.reference(public, cancel=cancel)), loaded)
    timer = QTimer(main)
    timer.setInterval(50)

    def finish(ok, error=""):
        state["phase"] = 3
        state["ok"] = ok
        (output / "result.json").write_text(json.dumps(dict(ok=ok, error=error,
            frozen=bool(getattr(sys, "frozen", False)), public_query=public["usi"],
            seconds=round(time.monotonic() - state["started"], 3)), indent=2))
        main.close()

    def tick():
        try:
            if state["phase"] == 3:
                if main.runtime.closed:
                    timer.stop()
                    app.exit(0 if state["ok"] else 1)
                return
            if time.monotonic() - state["started"] > 180:
                raise TimeoutError("Live GNPS smoke exceeded its deadline")
            if window.runtime.job is not None:
                return
            if state["phase"] == 0:
                assert state.get("loaded"), window.statusBar().currentMessage()
                window.search()
                state["phase"] = 1
            elif state["phase"] == 1:
                searches = window.analysis["searches"]
                assert set(searches) == {"standard", "analog"}
                assert not any(result.get("error") for result in searches.values()), searches
                assert any(hit["accession"] == public["accession"] and hit["score"] >= .99
                           for hit in searches["standard"]["hits"])
                if not window.analysis["references"]:
                    if window._active_hit and window._active_hit["usi"] in window._reference_attempts:
                        raise AssertionError(window.statusBar().currentMessage())
                    return
                assert any(reference["structure_png"] for reference in window.analysis["references"].values())
                path = output / "public-reference.kmzgnps"
                save_analysis(path, window.analysis)
                window.open_analysis(path)
                assert window.offline and len(window.mirror.listDataItems()) == 2
                window.grab().save(str(output / "gnps-public.png"))
                finish(True)
        except Exception as exc:
            finish(False, str(exc))

    timer.timeout.connect(tick)
    timer.start()
    main._gnps_live_timer = timer
