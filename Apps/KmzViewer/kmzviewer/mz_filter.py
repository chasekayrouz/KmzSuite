"""Exact, non-destructive ion chromatogram views and a bounded result cache."""
from collections import OrderedDict

import numpy as np

from kmzcore.cancel import Cancelled
from kmzviewer.extract import MzRange, _finite_max, _prepare, format_number
from kmzviewer.spectra import SpectrumReader, read_ahead, read_arrays


def mz_range(lower, upper):
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or lower < 0 or upper < lower:
        raise ValueError("Enter finite m/z values with 0 ≤ Start ≤ Stop.")
    return MzRange(lower, upper)


def range_label(bounds):
    return f"m/z {format_number(bounds.lower)}–{format_number(bounds.upper)}"


class FilterCache:
    """GUI-owned LRU of completed arrays; trace IDs never recur in a session."""
    def __init__(self, budget_bytes=16 * 1024 * 1024, limit=512):
        self.budget_bytes = budget_bytes
        self.limit = limit
        self.nbytes = 0
        self._items = OrderedDict()

    def get(self, trace_id, bounds):
        key = (trace_id, bounds)
        result = self._items.get(key)
        if result is not None:
            self._items.move_to_end(key)
        return result

    def put(self, trace_id, bounds, values):
        key = (trace_id, bounds)
        previous = self._items.pop(key, None)
        if previous is not None:
            self.nbytes -= previous.nbytes
        if values.nbytes > self.budget_bytes or self.limit <= 0:
            return
        self._items[key] = values
        self.nbytes += values.nbytes
        while self.nbytes > self.budget_bytes or len(self._items) > self.limit:
            _, old = self._items.popitem(last=False)
            self.nbytes -= old.nbytes

    def drop_traces(self, trace_ids):
        for key in list(self._items):
            if key[0] in trace_ids:
                self.nbytes -= self._items.pop(key).nbytes


def extract_filtered_values(path, scans, traces, bounds, *, cache=None,
                            on_progress=None, is_cancelled=None):
    """Read each needed scan once; retain EIC window order and multiplicity.

    Use native float64 slice reductions, never prefix subtraction, resampling or
    centroiding. Sorting/finite-mass preparation is shared by sums and maxima.
    ``None`` bounds is used for an unrestricted BPC extraction.
    """
    interval = (-np.inf, np.inf) if bounds is None else (bounds.lower, bounds.upper)
    windows, plans, output = {}, [], {}
    for trace in traces:
        if trace.optical:
            continue
        if trace.kind == "eic":
            if not trace.mz_windows:
                raise ValueError("EIC extraction windows are unavailable. Extract the EIC again.")
            clipped = [(max(lo, interval[0]), min(hi, interval[1])) for lo, hi in trace.mz_windows]
            clipped = [(lo, hi) for lo, hi in clipped if lo <= hi]
        elif trace.kind in {"tic", "bpc"}:
            clipped = [interval]
        else:
            raise ValueError(f"Unsupported ion chromatogram: {trace.kind}")
        values = output[trace.trace_id] = np.zeros(len(trace.row), dtype=np.float64)
        indices = [windows.setdefault(window, len(windows)) for window in clipped]
        if indices and len(trace.row):
            plans.append([trace, values, indices, 0])
    if not plans:
        return output
    mass_bounds = np.asarray(list(windows), dtype=np.float64)
    maximum_indices = {i for trace, _, indices, _ in plans if trace.kind == "bpc" for i in indices}
    sum_indices = {i for trace, _, indices, _ in plans if trace.kind != "bpc" for i in indices}
    rows = np.unique(np.concatenate([trace.row for trace, *_ in plans]))
    # Python ints and floats compare and add exactly as the NumPy scalars they replace.
    plan_rows = [trace.row.tolist() for trace, *_ in plans]
    with SpectrumReader(path) as reader:
        read_ahead(reader, cache, path, scans.spectrum_index[rows])
        for done, row in enumerate(rows.tolist(), 1):
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            index = int(scans.spectrum_index[row])
            mz, intensity = read_arrays(cache, path, index, lambda: reader.read(index))
            if is_cancelled is not None and is_cancelled():
                raise Cancelled()
            if mz.ndim != 1 or intensity.shape != mz.shape:
                raise ValueError("The m/z and intensity arrays must be equal-length vectors.")
            mz, intensity = _prepare(mz, intensity)
            native_max = _finite_max is not None and intensity.dtype == np.float64 and intensity.flags.c_contiguous
            left = np.searchsorted(mz, mass_bounds[:, 0], side="left")
            right = np.searchsorted(mz, mass_bounds[:, 1], side="right")
            sums = np.zeros(len(windows))
            maxima = np.zeros(len(windows))
            starts, stops = left.tolist(), right.tolist()
            for i in np.flatnonzero(left < right).tolist():
                chunk = intensity[starts[i]:stops[i]]
                if native_max and i in maximum_indices and i not in sum_indices:
                    # A nonzero maximum is one exact value; zero keeps NumPy's signed-zero choice.
                    count, top = _finite_max(chunk)
                    if top:
                        maxima[i] = top
                        continue
                    if not count:
                        continue
                finite = chunk[np.isfinite(chunk)]
                if finite.size:
                    if i in sum_indices:
                        sums[i] = np.sum(finite, dtype=np.float64)
                    if i in maximum_indices:
                        maxima[i] = np.max(finite)
            totals = sums.tolist()
            for plan, trace_rows in zip(plans, plan_rows):
                trace, values, indices, position = plan
                if position < len(trace_rows) and trace_rows[position] == row:
                    if trace.kind == "bpc":
                        values[position] = maxima[indices[0]]
                    else:
                        total = 0.0
                        for i in indices:
                            total += totals[i]
                        values[position] = total
                    plan[3] += 1
            if on_progress is not None and (done == len(rows) or done % 10 == 0):
                on_progress(done, len(rows))
    return output
