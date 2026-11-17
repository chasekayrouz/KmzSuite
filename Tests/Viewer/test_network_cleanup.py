"""Network helpers delete their Qt objects on the calling worker thread (offline).

Applications collect reference cycles only on the GUI thread, so a worker-thread
QTimer, QEventLoop or QNetworkAccessManager must never be left for that collector,
even when the helper raises and a traceback keeps its frame alive.
"""
import threading
from pathlib import Path

import pytest
import shiboken6
from PySide6 import QtNetwork
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from kmzviewer import dreams_client, gnps_client, reference_pack


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class Offline(QtNetwork.QNetworkAccessManager):
    """Answer every request from a data: URL, so no network service is contacted."""
    created = []

    def __init__(self, *args):
        super().__init__(*args)
        Offline.created.append(threading.get_ident())

    def createRequest(self, operation, request, data=None):
        request = QtNetwork.QNetworkRequest(QUrl("data:,offline"))
        return super().createRequest(QtNetwork.QNetworkAccessManager.Operation.GetOperation, request, None)


class FailingClose:
    """An output whose final flush fails on close(), as on a full disk."""

    def __init__(self, path):
        self._file = open(path, "wb")

    def write(self, data):
        return self._file.write(data)

    def close(self):
        self._file.close()
        raise OSError(28, "No space left on device")


DREAMS = "https://anton-bushuiev-dreams.hf.space/offline"


@pytest.mark.parametrize("call, created", [
    (lambda tmp: gnps_client.qt_request("https://external.gnps2.org/offline"), True),
    (lambda tmp: gnps_client.qt_request("https://external.gnps2.org/offline", payload={"x": float("nan")}), False),
    (lambda tmp: dreams_client.qt_http(DREAMS), True),
    (lambda tmp: dreams_client.qt_http(DREAMS, output=tmp / "missing" / "refs.bin"), False),
    (lambda tmp: dreams_client.qt_http(DREAMS, output=tmp / "close-fails"), True),
    (lambda tmp: reference_pack._download(reference_pack.HOST + reference_pack.LIBRARIES[0][0] + ".mgf", tmp / "pack"), True),
], ids=["gnps", "gnps-unencodable", "dreams", "dreams-unopenable", "dreams-close-fails", "references"])
def test_failed_request_deletes_qt_objects_on_its_worker_thread(call, created, tmp_path, monkeypatch, qapp):
    deleted, failures = [], []
    delete = shiboken6.delete

    def recorded(item):
        deleted.append((type(item).__name__, threading.get_ident()))
        delete(item)
    monkeypatch.setattr(shiboken6, "delete", recorded)
    monkeypatch.setattr(QtNetwork, "QNetworkAccessManager", Offline)
    monkeypatch.setattr(Offline, "created", [])
    open_path = Path.open
    monkeypatch.setattr(Path, "open", lambda self, mode="r", *args, **kwargs: FailingClose(self)
                        if self.name == "close-fails" else open_path(self, mode, *args, **kwargs))

    def worker():
        try:
            call(tmp_path)
        except Exception as exc:  # No HTTP status from a data: URL, or the injected failure.
            failures.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert not thread.is_alive() and len(failures) == 1
    # Every Qt object a helper creates is deleted on its worker thread, whatever failed.
    assert Offline.created == ([thread.ident] if created else [])
    assert deleted == ([(name, thread.ident) for name in ("QTimer", "QEventLoop", "Offline")] if created else [])
