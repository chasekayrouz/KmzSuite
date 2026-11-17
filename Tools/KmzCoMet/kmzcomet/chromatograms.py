"""On-demand, full-run summed EICs. Analysis measurements remain unchanged."""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kmzcore.cancel import Cancelled
from .engine import file_hash
from .sources import InputSource, eic_scans as sample_scans


@dataclass(frozen=True)
class EICData:
    rt: np.ndarray
    y: np.ndarray
    missing_coverage: int = 0
    unknown_coverage: int = 0

    @property
    def nbytes(self):
        return self.rt.nbytes + self.y.nbytes


@dataclass(frozen=True)
class EICEntry:
    feature_id: str
    sample_id: str
    filename: str
    native_sample_id: int | None
    mz: float
    polarity: int
    ppm: float
    unit: str
    data: EICData | None
    error: str = ''

    @property
    def title(self):
        stored = f' · stored {self.native_sample_id}' if self.native_sample_id is not None else ''
        sign = {1: '+', -1: '−', 0: '?'}[self.polarity]
        return f'{self.filename} · {self.sample_id}{stored} · {self.feature_id} · {self.mz:.6f} {sign}'


@dataclass(frozen=True)
class EICBatch:
    entries: tuple[EICEntry, ...]
    reused: int
    files_read: int


class ChromatogramCache:
    """One analysis/session, byte-bounded LRU; open tabs own their own references.

    A worker is the only caller while extraction runs. Source stat identities are
    checked on every request; new identities require the saved content hash.
    """
    def __init__(self, result, *, max_bytes=64 * 1024 * 1024):
        self.result = result
        self.max_bytes = max_bytes
        self.nbytes = 0
        self._data = OrderedDict()
        self._verified = {}

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _verify(self, sample, checkpoint):
        source = InputSource(sample.path, sample.native_sample_id)
        records = sample.source_files or [dict(path=str(sample.path), size_bytes=sample.size_bytes, sha256=sample.sha256)]
        if sample.source_files and [str(p) for p in source.files()] != [r['path'] for r in records]:
            raise ValueError('The acquisition sidecar set differs from the saved analysis')
        stamps = []
        for record in records:
            checkpoint()
            path = Path(record['path'])
            before = self._stamp(path)
            if before[2] != record['size_bytes']:
                raise ValueError(f'{path.name}: source differs from the saved analysis')
            if self._verified.get(path) != before:
                if file_hash(path, checkpoint) != record['sha256']:
                    raise ValueError(f'{path.name}: source differs from the saved analysis')
                if self._stamp(path) != before:
                    raise ValueError(f'{path.name}: source changed while being checked')
                self._verified[path] = before
            stamps.append((path, before))
        return stamps

    def _put(self, key, data):
        if data.nbytes > self.max_bytes:
            return
        if key in self._data:
            self.nbytes -= self._data.pop(key).nbytes
        while self._data and self.nbytes + data.nbytes > self.max_bytes:
            self.nbytes -= self._data.popitem(last=False)[1].nbytes
        self._data[key] = data
        self.nbytes += data.nbytes

    def extract(self, cells, *, progress=lambda message: None, checkpoint=lambda: None):
        """Cells are (feature index, sample index); no widgets or spectra retained."""
        result = self.result
        cells = tuple(dict.fromkeys(cells))
        grouped = defaultdict(list)
        for row, column in cells:
            if not (0 <= row < len(result.features) and 0 <= column < len(result.samples)):
                raise ValueError('EIC selection is outside this analysis')
            grouped[column].append(row)
        extracted, errors = {}, {}
        reused = files_read = 0
        for column, rows in grouped.items():
            sample = result.samples[column]
            checkpoint()
            progress(f'Checking EIC source {sample.id}: {sample.path.name}')
            keys = {row: (column, result.features[row].mz, result.features[row].polarity, result.settings.mz_ppm) for row in rows}
            try:
                stamps = self._verify(sample, checkpoint)
                ready = {}
                pending = {}
                for key in keys.values():
                    if key in self._data:
                        ready[key] = self._data[key]
                        self._data.move_to_end(key)
                    else:
                        pending[key] = key
                if pending:
                    files_read += 1
                    ready.update(self._read(sample, tuple(pending), progress, checkpoint))
                checkpoint()
                if any(self._stamp(path) != before for path, before in stamps):
                    raise ValueError('Source changed during EIC extraction; traces discarded')
                for row, key in keys.items():
                    extracted[row, column] = ready[key]
                    reused += key not in pending
                for key in pending:
                    self._put(key, ready[key])
            except Cancelled:
                raise
            except (OSError, ValueError, RuntimeError) as error:
                for row in rows:
                    errors[row, column] = str(error)
        checkpoint()
        entries = []
        for row, column in cells:
            feature, sample = result.features[row], result.samples[column]
            raw_unit = next(iter(sample.units))
            unit = 'counts' if raw_unit == 'number of detector counts' else raw_unit
            entries.append(EICEntry(feature.id, sample.id, sample.path.name, sample.native_sample_id,
                                    feature.mz, feature.polarity, result.settings.mz_ppm, unit,
                                    extracted.get((row, column)), errors.get((row, column), '')))
        return EICBatch(tuple(entries), reused, files_read)

    @staticmethod
    def _read(sample, keys, progress, checkpoint):
        by_polarity = defaultdict(list)
        for key in keys:
            by_polarity[key[2]].append(key)
        times = defaultdict(list)
        plans = {}
        for polarity, requested in by_polarity.items():
            masses = np.array([key[1] for key in requested])
            delta = requested[0][3] * 1e-6
            plans[polarity] = (masses * (1.0-delta), masses * (1.0+delta))
        values = {key: [] for key in keys}
        missing = dict.fromkeys(keys, 0)
        unknown = dict.fromkeys(keys, 0)
        for index, scan in enumerate(sample_scans(sample, checkpoint)):
            checkpoint()
            if index % 250 == 0:
                progress(f'Extract EICs {sample.id}: {index:,} / {sample.scans:,} scans')
            requested = by_polarity.get(scan.polarity, ())
            if not requested:
                continue
            times[scan.polarity].append(scan.rt)
            lows, highs = plans[scan.polarity]
            starts = np.searchsorted(scan.mz, lows, side='left')
            stops = np.searchsorted(scan.mz, highs, side='right')
            for key, lo, hi, start, stop in zip(requested, lows, highs, starts, stops):
                if scan.windows and not any(left <= lo and hi <= right for left, right in scan.windows):
                    values[key].append(np.nan)  # Unacquired is never a measured zero.
                    missing[key] += 1
                else:
                    values[key].append(float(np.sum(scan.intensity[start:stop], dtype=np.float64)))
                    unknown[key] += not bool(scan.windows)
        axes = {polarity: np.asarray(times[polarity], dtype=np.float64) for polarity in by_polarity}
        for axis in axes.values():
            axis.setflags(write=False)
        data = {}
        for key in keys:
            y = np.asarray(values[key], dtype=np.float64)
            y.setflags(write=False)
            data[key] = EICData(axes[key[2]], y, missing[key], unknown[key])
        return data
