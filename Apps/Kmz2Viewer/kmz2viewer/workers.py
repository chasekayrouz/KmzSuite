"""Background MS2 indexing."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal

from kmz2viewer.index import index_ms2
from kmzcore.cancel import Cancelled


class LoadWorker(QObject):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths
        self._cancel = Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        scans = []
        errors = []
        names = []
        paths = []
        try:
            total = len(self.paths)
            for number, path in enumerate(self.paths, start=1):
                if self._cancel.is_set():
                    raise Cancelled()
                name = Path(path).name

                def on_progress(done: int, file_total: int, name=name, number=number) -> None:
                    self.progress.emit(done, file_total, f"Reading {name} ({number}/{total})")

                try:
                    found = index_ms2(path, on_progress=on_progress, is_cancelled=self._cancel.is_set)
                except Cancelled:
                    raise
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                scans.extend(found)
                names.append(name)
                paths.append(path)
            if not scans and errors:
                self.failed.emit("\n".join(errors))
                return
            if self._cancel.is_set():
                raise Cancelled()
            self.succeeded.emit({"scans": scans, "errors": errors, "names": names, "paths": paths})
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
