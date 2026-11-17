"""Background work owns no widgets; publication wins over late cancellation."""
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from kmzcore.cancel import Cancelled
from kmzcomet.engine import analyze
from kmzcomet.history import load_analysis, save_analysis


@dataclass
class Completed:
    result: object
    path: Path | None
    save_error: str = ''


class Worker(QObject):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.stop = Event()

    def cancel(self):
        self.stop.set()

    def checkpoint(self):
        if self.stop.is_set():
            raise Cancelled()

    @Slot()
    def run(self):
        try:
            self.checkpoint()
            value = self.operation(self)
            # Operations return only after their commit point. Do not add a
            # cancellation check here that could hide a committed analysis.
            self.succeeded.emit(value)
        except Cancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.finished.emit()


def analysis_worker(inputs, settings, directory):
    def operation(worker):
        result = analyze(inputs, settings, progress=worker.progress.emit, checkpoint=worker.checkpoint)
        worker.checkpoint()
        worker.progress.emit('Saving analysis…')
        try:
            path = save_analysis(result, directory, checkpoint=worker.checkpoint)
            return Completed(result, path)
        except Cancelled:
            raise
        except Exception as error:
            return Completed(result, None, str(error))
    return Worker(operation)


def save_worker(result, directory):
    def operation(worker):
        path = save_analysis(result, directory, checkpoint=worker.checkpoint)
        return Completed(result, path)
    return Worker(operation)


def load_worker(path):
    def operation(worker):
        result = load_analysis(path)
        worker.checkpoint()
        return Completed(result, Path(path))
    return Worker(operation)
