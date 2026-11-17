"""In-memory chromatogram session. No GUI types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
from kmzcore.trace_style import chromatogram_palette


@dataclass
class DadScanTable:
    """Optical spectra have their own acquisition times and no MS polarity."""
    spectrum_index: np.ndarray
    rt_min: np.ndarray
    scan_id: list[str]

    def __len__(self):
        return len(self.rt_min)


@dataclass
class ScanTable:
    """MS1 scans only, in acquisition order. Arrays share one row index."""

    spectrum_index: np.ndarray
    rt_min: np.ndarray
    tic: np.ndarray
    polarity: np.ndarray
    centroid: np.ndarray
    scan_id: list[str]
    sparse_profile: list[dict | None] = field(default_factory=list)
    dad: DadScanTable | None = None
    dad_error: str = ""

    def __len__(self) -> int:
        return int(self.rt_min.shape[0])


@dataclass
class Trace:
    """One plotted chromatogram. ``row`` indexes the parent file's ScanTable."""

    trace_id: int
    file_id: int
    kind: str
    label: str
    rt: np.ndarray
    y: np.ndarray
    row: np.ndarray
    detail: str
    color: str
    visible: bool = True
    extraction_key: tuple = ()
    mz_windows: tuple[tuple[float, float], ...] = ()

    @property
    def group_key(self) -> str:
        # Identity uses full extraction parameters, never rounded display labels.
        return repr(self.extraction_key or (self.kind, self.label, self.detail))

    @property
    def group_label(self) -> str:
        if self.kind in {"eic", "bpc"} and self.extraction_key:
            polarity = {"all": "all polarities", "positive": "+", "negative": "−"}[self.extraction_key[1]]
            return f"{self.label} · {polarity}"
        return self.label

    @property
    def optical(self) -> bool:
        return self.kind in {"twc", "ewc"}

    @property
    def unit(self) -> str:
        return "mAU" if self.optical else "counts"


@dataclass
class LoadedFile:
    path: str
    name: str
    file_id: int
    scans: ScanTable
    traces: list[Trace] = field(default_factory=list)
    color: str = "#0072b2"

    def trace(self, trace_id: int) -> Trace | None:
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None


class Session:
    def __init__(self) -> None:
        self.files: dict[int, LoadedFile] = {}
        self._next_file_id = 1
        self._next_trace_id = 1
        self._color = 0
        self.emphasized: set[int] = set()
        self.grouping = "file"

    def __contains__(self, path: str) -> bool:
        target = str(Path(path).resolve())
        return any(str(Path(item.path).resolve()) == target for item in self.files.values())

    def next_color(self) -> str:
        palette = chromatogram_palette()
        color = palette[self._color % len(palette)]
        self._color += 1
        return color

    def new_trace_id(self) -> int:
        trace_id = self._next_trace_id
        self._next_trace_id += 1
        return trace_id

    def add_file(self, path: str, scans: ScanTable, tic: Trace) -> LoadedFile:
        file_id = self._next_file_id
        self._next_file_id += 1
        tic.file_id = file_id
        loaded = LoadedFile(
            path=str(path),
            name=Path(path).name,
            file_id=file_id,
            scans=scans,
            traces=[tic],
            color=tic.color,
        )
        self.files[file_id] = loaded
        return loaded

    def file_by_path(self, path: str) -> LoadedFile | None:
        target = str(Path(path).resolve())
        for loaded in self.files.values():
            if str(Path(loaded.path).resolve()) == target:
                return loaded
        return None

    def find_trace(self, trace_id: int) -> tuple[LoadedFile, Trace] | None:
        for loaded in self.files.values():
            trace = loaded.trace(trace_id)
            if trace is not None:
                return loaded, trace
        return None

    def ordered_traces(self, visible_only: bool = False) -> list[tuple[LoadedFile, Trace]]:
        rows = [(loaded, trace) for loaded in self.files.values() for trace in loaded.traces]
        if self.grouping == "type":
            rows = [row for _key, group in self.trace_groups() for row in group]
        return [row for row in rows if not visible_only or row[1].visible]

    def trace_groups(self) -> list[tuple[str, list[tuple[LoadedFile, Trace]]]]:
        groups = {}
        for loaded in self.files.values():
            for trace in loaded.traces:
                groups.setdefault(trace.group_key, []).append((loaded, trace))
        # Stable extraction order within each family; file sorting changes only
        # the order of sources inside a group, not the groups themselves.
        kinds = {"tic": 0, "bpc": 1, "eic": 2, "twc": 3, "ewc": 4}
        return sorted(groups.items(), key=lambda item: (
            kinds.get(item[1][0][1].kind, 5), min(t.trace_id for _, t in item[1])))

    def set_visible(self, trace_id: int, visible: bool) -> None:
        found = self.find_trace(trace_id)
        if found is not None:
            found[1].visible = visible

    def remove_trace(self, trace_id: int) -> None:
        self.remove_traces({trace_id})

    def remove_traces(self, trace_ids: set[int]) -> None:
        self.emphasized.difference_update(trace_ids)
        for loaded in self.files.values():
            loaded.traces = [trace for trace in loaded.traces if trace.trace_id not in trace_ids]

    def set_trace_colors(self, trace_ids: set[int], color: str) -> None:
        for _, trace in self.ordered_traces():
            if trace.trace_id in trace_ids:
                trace.color = color

    def move_files(self, file_ids: list[int], before: int) -> bool:
        order = list(self.files)
        chosen = set(file_ids)
        if not chosen or not chosen.issubset(order) or not 0 <= before <= len(order):
            return False
        moving = [fid for fid in order if fid in chosen]
        remaining = [fid for fid in order if fid not in chosen]
        destination = sum(fid not in chosen for fid in order[:before])
        result = remaining[:destination] + moving + remaining[destination:]
        return self._set_file_order(result)

    def sort_files(self) -> bool:
        def key(fid):
            # Case-insensitive natural order: Run 2 before Run 10; ties stay stable.
            return tuple((1, int(part)) if part.isdigit() else (0, part)
                         for part in re.split(r"([0-9]+)", self.files[fid].name.casefold()))
        return self._set_file_order(sorted(self.files, key=key))

    def _set_file_order(self, order: list[int]) -> bool:
        if order == list(self.files):
            return False
        self.files = {fid: self.files[fid] for fid in order}
        return True

    def remove_file(self, file_id: int) -> None:
        loaded = self.files.pop(file_id, None)
        if loaded:
            self.emphasized.difference_update(trace.trace_id for trace in loaded.traces)

    def set_file_color(self, file_id: int, color: str) -> None:
        loaded = self.files.get(file_id)
        if loaded is not None:
            loaded.color = color
            for trace in loaded.traces:
                trace.color = color

    def add_traces(self, file_id: int, traces: list[Trace]) -> None:
        loaded = self.files.get(file_id)
        if loaded is not None:
            for trace in traces:
                trace.file_id = file_id
                trace.color = loaded.color
            loaded.traces.extend(traces)

    def toggle_emphasis(self, trace_id: int) -> None:
        if self.find_trace(trace_id) is None:
            return
        if trace_id in self.emphasized:
            self.emphasized.remove(trace_id)
        else:
            self.emphasized.add(trace_id)

    def has_tic(self, file_id: int) -> bool:
        loaded = self.files.get(file_id)
        if loaded is None:
            return False
        return any(trace.kind == "tic" for trace in loaded.traces)
