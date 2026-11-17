"""Background indexing, extraction, and spectrum reads."""

from __future__ import annotations

import queue
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, Signal

from kmzcore.cancel import Cancelled
from kmzcore.sparse_profile import rendering_connections
from kmzviewer.extract import extract_bpc, extract_eics, parse_targets, rows_for_polarity, trace_plan
from kmzviewer.mz_filter import extract_filtered_values
from kmzviewer.mzml_index import load_ms1
from kmzviewer.profile_rendering import zero_baseline_curve
from kmzviewer.spectra import ArrayCache, SpectrumReader, read_arrays
from kmzviewer.dad import extract_wavelengths


class IndexWorker(QObject):
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
        loaded = []
        errors = []
        total_files = len(self.paths)
        try:
            for number, path in enumerate(self.paths, start=1):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                name = Path(path).name

                def on_progress(done: int, total: int, name=name, number=number) -> None:
                    self.progress.emit(done, total, f"Reading {name} ({number}/{total_files})")

                try:
                    table = load_ms1(path, on_progress=on_progress, is_cancelled=self._cancel.is_set)
                except Cancelled:
                    self.cancelled.emit()
                    return
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                loaded.append((path, table))
            if not loaded and errors:
                self.failed.emit("\n".join(errors))
                return
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.succeeded.emit({"files": loaded, "errors": errors})
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ExtractWorker(QObject):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, jobs: list[dict], cache: ArrayCache | None = None) -> None:
        super().__init__()
        self.jobs = jobs
        self._cache = cache
        self._cancel = Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            target = self.jobs[0].get("target") if self.jobs else None
            results = []
            scan_total = 0
            counts = []
            for job in self.jobs:
                count = len(job['scans']) if 'wavelengths' in job else int(rows_for_polarity(job["scans"], job["polarity"]).shape[0])
                counts.append(count)
                scan_total += count
            finished = 0
            for job, count in zip(self.jobs, counts):
                if self._cancel.is_set():
                    self.cancelled.emit()
                    return
                name = job["name"]

                def on_progress(done: int, _total: int, base=finished, name=name) -> None:
                    self.progress.emit(base + done, max(scan_total, 1), f"Extracting {name}")

                if job.get('bpc'):
                    traces = [extract_bpc(job['path'], job['scans'], job['polarity'], job['trace_ids'][0], job['colors'][0],
                                          cache=self._cache, on_progress=on_progress, is_cancelled=self._cancel.is_set)]
                elif 'wavelengths' in job:
                    traces = extract_wavelengths(job['path'], job['scans'], job['wavelengths'], job['trace_ids'], job['color'],
                                                cache=self._cache, on_progress=on_progress, is_cancelled=self._cancel.is_set)
                else:
                    traces = extract_eics(
                        job["path"], job["scans"], target, job["polarity"], job["merge"],
                        job["trace_ids"], job["colors"], on_progress=on_progress,
                        is_cancelled=self._cancel.is_set, cache=self._cache,
                    )
                for trace in traces:
                    trace.file_id = job["file_id"]
                results.append((job["file_id"], traces))
                finished += count
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.succeeded.emit(results)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


def plan_jobs(session_files: list, text: str, ppm: float, polarity: str, merge: bool, new_id) -> list[dict]:
    """Reserve trace IDs and snapshot each source's current display color."""
    target = parse_targets(text, ppm)
    plan = trace_plan(target, polarity, merge)
    jobs = []
    for loaded in session_files:
        jobs.append(
            {
                "path": loaded.path,
                "name": loaded.name,
                "file_id": loaded.file_id,
                "scans": loaded.scans,
                "target": target,
                "polarity": polarity,
                "merge": merge,
                "trace_ids": [new_id() for _ in plan],
                "colors": [loaded.color for _ in plan],
            }
        )
    return jobs


def plan_bpc_jobs(session_files, polarity, new_id):
    return [dict(path=f.path, name=f.name, file_id=f.file_id, scans=f.scans,
                 bpc=True, polarity=polarity, trace_ids=[new_id()], colors=[f.color]) for f in session_files]


class FilterWorker(ExtractWorker):
    def __init__(self, jobs, bounds, cache=None):
        super().__init__(jobs, cache)
        self.bounds = bounds

    def run(self):
        try:
            results = {}
            total = sum(len(job['scans']) for job in self.jobs)
            finished = 0
            for job in self.jobs:
                if self._cancel.is_set():
                    raise Cancelled()

                def progress(done, count, base=finished, job=job):
                    # A polarity subset may visit fewer rows than its source.
                    scaled = int(len(job['scans']) * done / max(count, 1))
                    self.progress.emit(base + scaled, max(total, 1), f"Filtering {job['name']}")

                results.update(extract_filtered_values(job['path'], job['scans'], job['traces'], self.bounds,
                    cache=self._cache, on_progress=progress, is_cancelled=self._cancel.is_set))
                finished += len(job['scans'])
            if self._cancel.is_set():
                raise Cancelled()
            self.succeeded.emit(results)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class SpectrumWorker(QObject):
    result = Signal(int, object)
    batch_result = Signal(int, object)

    def __init__(self, cache: ArrayCache | None = None) -> None:
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self._stopping = Event()
        self._readers: dict[str, SpectrumReader] = {}
        self._cache = cache if cache is not None else ArrayCache()

    def request(self, token: int, path: str, spectrum_index: int) -> None:
        if not self._stopping.is_set():
            self._queue.put(("read", token, path, int(spectrum_index)))

    def close_path(self, path: str) -> None:
        self._queue.put(("close", path))

    def request_batch(self, token: int, scans: list[tuple[int, str, int, object]]) -> None:
        """Decode a complete comparison frame; per-file errors remain independent."""
        if not self._stopping.is_set():
            self._queue.put(("batch", token, tuple(scans)))

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(("stop",))

    def run(self) -> None:
        try:
            while not self._stopping.is_set():
                job = self._coalesce(self._queue.get())
                if job is None or job[0] == "stop":
                    return
                if job[0] == "close":
                    self._drop(job[1])
                    continue
                if job[0] == "idle":
                    continue
                # Each job's arrays end with its method's frame, so a thread
                # waiting for work never keeps closed files' spectra alive.
                if job[0] == "batch":
                    self._batch(job[1], job[2])
                else:
                    self._read_one(job[1], job[2], job[3])
        finally:
            self._close_all()

    def _batch(self, token: int, scans) -> None:
        results = {}
        for file_id, path, index, sparse in scans:
            if self._stopping.is_set():
                return
            try:
                mz, intensity = self._read(path, index)
                if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape:
                    raise ValueError("The m/z and intensity arrays must be equal-length vectors.")
                connections = rendering_connections(sparse, mz, intensity)
                curve = zero_baseline_curve(mz, intensity, connections)
                results[file_id] = (mz, intensity, connections, curve)
            except Exception as exc:
                results[file_id] = exc
        if not self._stopping.is_set():
            self.batch_result.emit(token, results)

    def _read_one(self, token: int, path: str, spectrum_index: int) -> None:
        try:
            arrays = self._read(path, spectrum_index)
            if not self._stopping.is_set():
                self.result.emit(token, {"mz": arrays[0], "intensity": arrays[1]})
        except Exception as exc:
            if not self._stopping.is_set():
                self.result.emit(token, exc)

    def _coalesce(self, job):
        if job[0] not in {"read", "batch"}:
            return job
        latest = job
        while True:
            try:
                nxt = self._queue.get_nowait()
            except queue.Empty:
                return latest if latest is not None else ("idle",)
            if nxt[0] == "stop":
                return None
            if nxt[0] == "close":
                self._drop(nxt[1])
                if latest is not None and (latest[0] == "read" and latest[2] == nxt[1]
                        or latest[0] == "batch" and any(scan[1] == nxt[1] for scan in latest[2])):
                    latest = None
                continue
            latest = nxt

    def _read(self, path: str, spectrum_index: int):
        def read() -> tuple:
            reader = self._readers.get(path)
            if reader is None:
                reader = SpectrumReader(path)
                self._readers[path] = reader
            return reader.read(spectrum_index)

        return read_arrays(self._cache, path, spectrum_index, read)

    def _drop(self, path: str) -> None:
        reader = self._readers.pop(path, None)
        if reader is not None:
            reader.close()
        self._cache.drop_path(path)

    def _close_all(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
