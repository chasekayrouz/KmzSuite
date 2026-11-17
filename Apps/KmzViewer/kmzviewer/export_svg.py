"""Editable, flat SVG geometry shared by the preview and file export."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
import math
import re
import threading
import xml.etree.ElementTree as ET

import numpy as np
# Matplotlib is imported where used: a first launch can spend seconds discovering
# fonts, which must not delay the window before any SVG is requested.
from kmzcore.trace_style import FILL_ALPHA
from kmzcore.ranges import fill_floor
from kmzcore.labels import select_peak_labels, format_mz_label
try:  # Optional exact "%.10g" path formatter; see _native.c.
    from kmzviewer._native import path_data as _native_path_data
except ImportError:
    _native_path_data = None


@dataclass
class ExportTrace:
    filename: str
    label: str
    color: str
    rt: np.ndarray
    y: np.ndarray
    unit: str = 'counts'
    centroid: bool = False
    mz_filter: tuple[float, float] | None = None


@dataclass
class ExportOptions:
    # Retain the original positional signature for existing scripted exports.
    # Layout/background and the old label flags no longer affect output.
    layout: str = "stacked"
    width_in: float = 7.2
    height_in: float = 4.8
    linewidth: float = 2.0
    fontsize: float = 12.0
    xmin: float = 0.0
    xmax: float = 1.0
    auto_y: bool = True
    ymin: float | None = None
    ymax: float | None = None
    filename_labels: bool = True
    trace_labels: bool = True
    transparent: bool = True
    tick_fontsize: float = 10.0
    spacing: float | None = None
    offset: float = 0.0
    x_tick_interval: float = 1.0
    y_tick_interval: float | None = None
    legend: bool = True
    fill: bool = True
    opacity: float = round(255 * FILL_ALPHA) / 255
    grid: bool = False
    grid_interval: float = 1.0
    legend_fontsize: float | None = None
    x_label: str = 'Retention time (min)'
    x_unit: str = 'min'
    right_auto_y: bool = True
    right_ymin: float | None = None
    right_ymax: float | None = None
    right_spacing: float | None = None
    right_y_tick_interval: float | None = None
    spectrum: bool = False


def _number(value):
    return format(float(value), ".10g")


def _path_data(xs, ys):
    """Path data for screen arrays: "%.10g" formats each float exactly as _number."""
    values = np.empty(2 * len(xs))
    values[0::2], values[1::2] = xs, ys
    if _native_path_data is not None:
        return _native_path_data(values)
    return "M " + ("%.10g %.10g L " * len(xs))[:-3] % tuple(values.tolist())


def scientific(value):
    """Every Y tick, including zero, uses an explicit power of ten."""
    mantissa, exponent = f"{value:.6e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+03d}"


def suggested_y_interval(low, high):
    from matplotlib.ticker import MaxNLocator
    values = MaxNLocator(nbins=4).tick_values(low, high)
    return float(values[1] - values[0])


def ticks(low, high, interval, start=0.0, *, kind="major ticks"):
    if not math.isfinite(interval) or interval <= 0 or not math.isfinite(start):
        raise ValueError("Intervals must be positive and finite.")
    first = math.ceil((low - start) / interval)
    last = math.floor((high - start) / interval)
    if last - first > 200:
        raise ValueError(f"Too many {kind}; increase the interval.")
    values = start + np.arange(first, last + 1) * interval
    return [float(v) for v in values if low <= v <= high]


def legend_label(trace):
    name = re.sub(r"(?:\.sp)?\.mzml$", "", trace.filename, flags=re.IGNORECASE)
    label = re.sub(r"\s*(?:±|\+/-)\s*[^\s]+\s*ppm\b", "", trace.label, flags=re.IGNORECASE).strip()
    label = re.sub(r"^(TIC|BPC)\s+MS1\b", r"\1", label, flags=re.IGNORECASE)
    return f"{name} ({label})"


def visible_runs(trace, xmin, xmax, shift=0.0):
    """Crop piecewise-linear runs at RT boundaries, retaining nonfinite gaps."""
    x = np.asarray(trace.rt, dtype=float) + shift
    y = np.asarray(trace.y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Chromatogram time and intensity arrays must match.")
    if np.any(np.diff(x[np.isfinite(x)]) < 0):
        raise ValueError("Chromatogram times must be ordered.")
    if trace.centroid:
        return [[(float(mass), 0.), (float(mass), float(peak))] for mass, peak in zip(x, y)
                if np.isfinite(mass) and np.isfinite(peak) and xmin <= mass <= xmax]
    finite = np.isfinite(x) & np.isfinite(y)
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    runs = []
    for begin, end in boundaries.reshape(-1, 2):
        rx, ry = x[begin:end], y[begin:end]
        if rx[-1] < xmin or rx[0] > xmax:
            continue
        left = int(np.searchsorted(rx, xmin, side="left"))
        right = int(np.searchsorted(rx, xmax, side="right"))
        points = list(zip(rx[left:right].tolist(), ry[left:right].tolist()))
        if 0 < left < len(rx) and rx[left] > xmin:
            weight = (xmin - rx[left-1]) / (rx[left] - rx[left-1])
            points.insert(0, (xmin, float(ry[left-1] + weight * (ry[left] - ry[left-1]))))
        if 0 < right < len(rx) and rx[right-1] < xmax:
            weight = (xmax - rx[right-1]) / (rx[right] - rx[right-1])
            points.append((xmax, float(ry[right-1] + weight * (ry[right] - ry[right-1]))))
        if points:
            runs.append(points)
    return runs


def stack_geometry(traces, options):
    levels = [0] * len(traces) if options.layout == 'overlay' else list(range(len(traces) - 1, -1, -1))
    runs = [visible_runs(trace, options.xmin, options.xmax, level * options.offset)
            for level, trace in zip(levels, traces)]
    peak = max((y for parts in runs for part in parts for _, y in part), default=0.0)
    low_peak = min((y for parts in runs for part in parts for _, y in part), default=0.0)
    spacing = options.spacing if options.spacing is not None else 1.1 * (max(0.0, peak) - min(0.0, low_peak))
    baselines = [level * spacing for level in levels]
    values = [0.0, *baselines]
    values.extend(y + base for parts, base in zip(runs, baselines) for part in parts for _, y in part)
    low, high = min(values), max(values)
    span = high - low or 1.0
    bounds = (low, high + span * .08)
    if not options.auto_y:
        bounds = (options.ymin, options.ymax)
    return runs, baselines, spacing, bounds


def plot_geometry(traces, options):
    """Separate native-unit axes; mixed stacks share screen-space baselines."""
    units = {trace.unit for trace in traces}
    if len(units) <= 1:
        runs, baselines, spacing, bounds = stack_geometry(traces, options)
        unit = next(iter(units), 'counts')
        return runs, baselines, {unit: spacing}, {unit: bounds}
    if units != {'counts', 'mAU'}:
        raise ValueError('Unsupported mixed export units.')
    levels = [0] * len(traces) if options.layout == 'overlay' else list(range(len(traces) - 1, -1, -1))
    runs = [visible_runs(trace, options.xmin, options.xmax, level * options.offset) for trace, level in zip(traces, levels)]
    spacing = {}
    for unit, override in [('counts', options.spacing), ('mAU', options.right_spacing)]:
        values = [y for trace, parts in zip(traces, runs) if trace.unit == unit for part in parts for _, y in part]
        span = max([0., *values]) - min([0., *values])
        spacing[unit] = override if override is not None else 1.1 * span
        if not math.isfinite(spacing[unit]) or spacing[unit] < 0:
            raise ValueError('Stack spacing must be nonnegative and finite.')
    # Use one step per table row on both scales. A zero trace has no physical
    # scale to match; the display fallback is explicit and does not alter data.
    scale = {unit: value or 1. for unit, value in spacing.items()}
    baselines = [level * spacing[trace.unit] for trace, level in zip(traces, levels)]
    values = [0.] + [base / scale[trace.unit] for trace, base in zip(traces, baselines)]
    values.extend((y + base) / scale[trace.unit] for trace, parts, base in zip(traces, runs, baselines) for part in parts for _, y in part)
    low, high = min(values), max(values)
    high += (high - low or 1.) * .08
    bounds = {unit: (low * step, high * step) for unit, step in scale.items()}
    if options.layout == 'overlay':
        for unit in units:
            values = [0.] + [y for trace, parts in zip(traces, runs) if trace.unit == unit for part in parts for _, y in part]
            low, high = min(values), max(values)
            bounds[unit] = (low, high + (high-low or 1.) * .08)
    if not options.auto_y:
        bounds['counts'] = (options.ymin, options.ymax)
    if not options.right_auto_y:
        bounds['mAU'] = (options.right_ymin, options.right_ymax)
    return runs, baselines, spacing, bounds


_OPTION_NAMES = tuple(field.name for field in fields(ExportOptions))


def _python_numbers(options):
    """Whether every option is non-numeric, a Python float or an int exact in float64.

    The array paths below repeat the scalar arithmetic exactly only for these.
    NumPy scalars and other numeric types promote differently, so such options
    render through the original point geometry and coordinates instead.
    """
    for name in _OPTION_NAMES:
        value = getattr(options, name, None)
        if type(value) is int:
            if abs(value) > 2 ** 53:
                return False
        elif type(value) not in (float, bool, str, type(None)):
            return False
    return True


def _runs(trace, xmin, xmax, shift=0.0):
    """visible_runs for array rendering. Curve runs are (x, y, head, tail): float arrays and
    the X bound object at each interpolated end, else None. Sticks stay point lists."""
    x = np.asarray(trace.rt, dtype=float) + shift
    y = np.asarray(trace.y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Chromatogram time and intensity arrays must match.")
    if np.any(np.diff(x[np.isfinite(x)]) < 0):
        raise ValueError("Chromatogram times must be ordered.")
    if trace.centroid:
        keep = np.isfinite(x) & np.isfinite(y) & (xmin <= x) & (x <= xmax)
        return [[(mass, 0.), (mass, peak)] for mass, peak in zip(x[keep].tolist(), y[keep].tolist())]
    finite = np.isfinite(x) & np.isfinite(y)
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    runs = []
    for begin, end in boundaries.reshape(-1, 2):
        rx, ry = x[begin:end], y[begin:end]
        if rx[-1] < xmin or rx[0] > xmax:
            continue
        left = int(np.searchsorted(rx, xmin, side="left"))
        right = int(np.searchsorted(rx, xmax, side="right"))
        xs, ys, head, tail = rx[left:right], ry[left:right], None, None
        if 0 < left < len(rx) and rx[left] > xmin:
            weight = (xmin - rx[left-1]) / (rx[left] - rx[left-1])
            xs, ys, head = np.r_[xmin, xs], np.r_[float(ry[left-1] + weight * (ry[left] - ry[left-1])), ys], xmin
        if 0 < right < len(rx) and rx[right-1] < xmax:
            weight = (xmax - rx[right-1]) / (rx[right] - rx[right-1])
            xs, ys, tail = np.r_[xs, xmax], np.r_[ys, float(ry[right-1] + weight * (ry[right] - ry[right-1]))], xmax
        if xs.size:
            runs.append((xs, ys, head, tail))
    return runs


def _ys(parts):
    """All Y values of one trace's runs, in drawing order."""
    if parts and isinstance(parts[0], list):
        return np.array([y for part in parts for _, y in part], dtype=float)
    return np.concatenate([part[1] for part in parts]) if parts else np.empty(0)


def _extremes(prefix, arrays):
    """min/max of [*prefix, *arrays] exactly as Python's builtins (first extreme wins)."""
    values = np.concatenate([np.asarray(prefix, dtype=float), *arrays])
    if not values.size:
        return 0.0, 0.0
    if np.isnan(values).any():
        flat = [*prefix, *(v for a in arrays for v in a.tolist())]
        return min(flat), max(flat)
    pick = lambda i: prefix[i] if i < len(prefix) else float(values[i])
    return pick(int(values.argmin())), pick(int(values.argmax()))


def _levels(traces, options):
    return [0] * len(traces) if options.layout == 'overlay' else list(range(len(traces) - 1, -1, -1))


def _stack_geometry(traces, options):
    levels = _levels(traces, options)
    runs = [_runs(trace, options.xmin, options.xmax, level * options.offset)
            for level, trace in zip(levels, traces)]
    ys = [_ys(parts) for parts in runs]
    low_peak, peak = _extremes([], ys)
    spacing = options.spacing if options.spacing is not None else 1.1 * (max(0.0, peak) - min(0.0, low_peak))
    baselines = [level * spacing for level in levels]
    low, high = _extremes([0.0, *baselines], [y + base for y, base in zip(ys, baselines)])
    span = high - low or 1.0
    bounds = (low, high + span * .08)
    if not options.auto_y:
        bounds = (options.ymin, options.ymax)
    return runs, baselines, spacing, bounds


def _plot_geometry(traces, options):
    units = {trace.unit for trace in traces}
    if len(units) <= 1:
        runs, baselines, spacing, bounds = _stack_geometry(traces, options)
        unit = next(iter(units), 'counts')
        return runs, baselines, {unit: spacing}, {unit: bounds}
    if units != {'counts', 'mAU'}:
        raise ValueError('Unsupported mixed export units.')
    levels = _levels(traces, options)
    runs = [_runs(trace, options.xmin, options.xmax, level * options.offset) for trace, level in zip(traces, levels)]
    ys = [_ys(parts) for parts in runs]
    spacing = {}
    for unit, override in [('counts', options.spacing), ('mAU', options.right_spacing)]:
        low, high = _extremes([0.], [y for trace, y in zip(traces, ys) if trace.unit == unit])
        spacing[unit] = override if override is not None else 1.1 * (high - low)
        if not math.isfinite(spacing[unit]) or spacing[unit] < 0:
            raise ValueError('Stack spacing must be nonnegative and finite.')
    # Use one step per table row on both scales. A zero trace has no physical
    # scale to match; the display fallback is explicit and does not alter data.
    scale = {unit: value or 1. for unit, value in spacing.items()}
    baselines = [level * spacing[trace.unit] for trace, level in zip(traces, levels)]
    low, high = _extremes([0.] + [base / scale[trace.unit] for trace, base in zip(traces, baselines)],
                          [(y + base) / scale[trace.unit] for trace, y, base in zip(traces, ys, baselines)])
    high += (high - low or 1.) * .08
    bounds = {unit: (low * step, high * step) for unit, step in scale.items()}
    if options.layout == 'overlay':
        for unit in units:
            low, high = _extremes([0.], [y for trace, y in zip(traces, ys) if trace.unit == unit])
            bounds[unit] = (low, high + (high-low or 1.) * .08)
    if not options.auto_y:
        bounds['counts'] = (options.ymin, options.ymax)
    if not options.right_auto_y:
        bounds['mAU'] = (options.right_ymin, options.right_ymax)
    return runs, baselines, spacing, bounds


def _crop_polygon(points, ymin, ymax):
    for bound, keep_above in ((ymin, True), (ymax, False)):
        output = []
        if not points:
            break
        prev = points[-1]
        for point in points:
            inside = point[1] >= bound if keep_above else point[1] <= bound
            was_inside = prev[1] >= bound if keep_above else prev[1] <= bound
            if inside != was_inside:
                fraction = (bound - prev[1]) / (point[1] - prev[1])
                output.append((prev[0] + fraction * (point[0] - prev[0]), bound))
            if inside:
                output.append(point)
            prev = point
        points = output
    return points


def _crop_line(points, ymin, ymax):
    segments, segment = [], []
    for a, b in zip(points, points[1:]):
        if max(a[1], b[1]) < ymin or min(a[1], b[1]) > ymax:
            if segment:
                segments.append(segment)
                segment = []
            continue
        endpoints = []
        for p, q in ((a, b), (b, a)):
            y = min(ymax, max(ymin, p[1]))
            x = p[0] if p[1] == q[1] else p[0] + (q[0]-p[0]) * (y-p[1]) / (q[1]-p[1])
            endpoints.append((x, y))
        if segment and segment[-1] != endpoints[0]:
            segments.append(segment)
            segment = []
        if not segment:
            segment.append(endpoints[0])
        segment.append(endpoints[1])
    if segment:
        segments.append(segment)
    return segments


def grid_segments(xmin, xmax, ymin, ymax, spacing, offset, interval):
    """Lines x = origin + y * offset/spacing align corresponding RT baselines."""
    if spacing == 0 and offset != 0:
        raise ValueError("Angled gridlines need positive stack spacing; increase spacing or disable gridlines.")
    slope = offset / spacing if spacing else 0.0
    shifts = (ymin * slope, ymax * slope)
    origins = ticks(xmin - max(shifts), xmax - min(shifts), interval, kind="gridlines")
    lines = []
    for origin in origins:
        low, high = ymin, ymax
        if slope:
            crossing = sorted(((xmin-origin) / slope, (xmax-origin) / slope))
            low, high = max(low, crossing[0]), min(high, crossing[1])
        if low <= high:
            lines.append(((origin + low*slope, low), (origin + high*slope, high)))
    return lines


_LABEL_WIDTHS = OrderedDict()  # (legend_font_properties key, label): unit-size width, least recent first.
_LABEL_WIDTH_LIMIT = 4096
_LABEL_WIDTH_LOCK = threading.Lock()


def legend_font_properties():
    """Unit-size legend FontProperties and a key for everything its text widths depend on.

    The key holds the effective properties (current rcParams defaults included),
    the font files they resolve to and the rcParams TextToPath's fonts read.
    """
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties, fontManager
    properties = FontProperties(family=["Helvetica", "Arial", "DejaVu Sans"], size=1)
    key = (tuple(properties.get_family()), properties.get_style(), properties.get_variant(),
           properties.get_weight(), properties.get_stretch(), properties.get_size(), properties.get_file(),
           tuple(fontManager._find_fonts_by_props(properties)),
           *(mpl.rcParams.get(name) for name in ("text.kerning_factor", "text.hinting_factor", "font.enable_last_resort")))
    return properties, key


def legend_layout(traces, options):
    """Full names in one right-hand column; automatic type is capped at 10 pt."""
    labels = [legend_label(trace) for trace in traces]
    properties, font_key = legend_font_properties()
    metrics, widths = None, []
    for label in labels:
        key = font_key, label
        with _LABEL_WIDTH_LOCK:
            width = _LABEL_WIDTHS.pop(key, None)
        if width is None:
            if metrics is None:
                from matplotlib.textpath import TextToPath
                metrics = TextToPath()
            width = metrics.get_text_width_height_descent(label, properties, False)[0]
        with _LABEL_WIDTH_LOCK:
            _LABEL_WIDTHS[key] = width
            while len(_LABEL_WIDTHS) > _LABEL_WIDTH_LIMIT:
                _LABEL_WIDTHS.popitem(last=False)
        widths.append(width)
    # Leave a little room for font substitution/rounding in SVG consumers.
    unit_width = max(widths, default=0) * 1.05
    available_height = options.height_in * 72 - options.fontsize * 3.4 - options.tick_fontsize * 2
    font = options.legend_fontsize
    if font is None:
        text_width = options.width_in * 72 * .30 - 24
        fit = min(10., text_width / max(unit_width, 1),
                  available_height / max(len(labels) * 1.55, 1))
        font = math.floor(fit * 100) / 100
    if not math.isfinite(font) or font < .1:
        raise ValueError("Legend font size must be at least 0.1 pt; widen the canvas or hide the legend if full names cannot fit.")
    if len(labels) * font * 1.55 > available_height:
        raise ValueError("Increase canvas height, reduce legend font size or hide the legend to fit every entry.")
    return font, unit_width * font + 24, labels


def render_chromatograms(traces: list[ExportTrace], options: ExportOptions) -> bytes:
    return render_with_layout(traces, options)[0]


def render_with_layout(traces, options):
    """SVG bytes plus the legend font, spacings and Y bounds they were drawn with."""
    from matplotlib.colors import to_hex, to_rgba
    if not traces:
        raise ValueError("Choose at least one chromatogram.")
    for value in (options.width_in, options.height_in, options.linewidth, options.fontsize, options.tick_fontsize):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Dimensions, line weights and font sizes must be positive and finite.")
    if not all(math.isfinite(v) for v in (options.xmin, options.xmax, options.offset)) or options.xmax <= options.xmin:
        raise ValueError("The retention-time range is empty or invalid.")
    if options.spacing is not None and (not math.isfinite(options.spacing) or options.spacing < 0):
        raise ValueError("Stack spacing must be nonnegative and finite.")
    if not math.isfinite(options.opacity) or not 0 <= options.opacity <= 1:
        raise ValueError("Fill opacity must be between 0 and 100%.")
    arrays = _python_numbers(options)
    runs, baselines, spacings, bounds = (_plot_geometry if arrays else plot_geometry)(traces, options)
    primary = 'counts' if 'counts' in bounds else 'mAU'
    mixed = len(bounds) > 1
    spacing = spacings[primary]
    ymin, ymax = bounds[primary]
    for low, high in bounds.values():
        if low is None or high is None or not all(math.isfinite(v) for v in (low, high)) or high <= low:
            raise ValueError("The Y range is empty or invalid.")
    xt = ticks(options.xmin, options.xmax, options.x_tick_interval)
    y_interval = options.y_tick_interval if options.y_tick_interval is not None else suggested_y_interval(ymin, ymax)
    yt = ticks(ymin, ymax, y_interval)
    width, height = options.width_in * 72, options.height_in * 72
    tf, font = options.tick_fontsize, options.fontsize
    left = max(62, max((len(scientific(v)) for v in yt), default=1) * tf * .62 + font * 2.2)
    right = max(22, max((len(_number(v)) for v in xt), default=1) * tf * .31 + 6)
    right_axis = mixed or (primary == 'mAU' and options.layout == 'overlay')
    right_ticks = []
    if right_axis:
        lower, upper = bounds['mAU']
        interval = options.right_y_tick_interval if mixed else options.y_tick_interval
        right_ticks = ticks(lower, upper, interval if interval is not None else suggested_y_interval(lower, upper))
        right = max(right, max((len(scientific(v)) for v in right_ticks), default=1) * tf * .62 + font * 2.2)
    bottom = font * 2 + tf * 2
    top = font * 1.4
    ph = height - top - bottom
    if ph < 72:
        raise ValueError("Increase canvas height or reduce axis font sizes to leave room for the plot.")
    legend_font, legend_width, labels = legend_layout(traces, options) if options.legend else (10, 0, [])
    legend_gap = max(18, right) if labels else 0
    pw = width - left - right - legend_width - legend_gap
    if pw < 72:
        raise ValueError("Increase canvas width, reduce font sizes or hide the legend to leave room for the plot.")
    root = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "version": "1.1",
        "width": f"{_number(options.width_in)}in", "height": f"{_number(options.height_in)}in",
        "viewBox": f"0 0 {_number(width)} {_number(height)}"})
    filters = list(dict.fromkeys(trace.mz_filter for trace in traces if trace.mz_filter is not None))
    measurement_text = "Native-unit traces" if filters else "Original measurements"
    description = (f"{measurement_text}; separate native-unit axes {', '.join(bounds)}; "
        f"vertical spacing {spacings}; bottom-fixed successive X offset {_number(options.offset)} {options.x_unit}.")
    if filters:
        description += " Display filter: " + "; ".join(f"m/z {_number(lo)}–{_number(hi)} inclusive" for lo, hi in filters) + "."
    ET.SubElement(root, "desc").text = description

    def element(tag, **attrs):
        return ET.SubElement(root, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})

    def text(x, y, content, size=font, anchor="start", **attrs):
        node = element("text", x=_number(x), y=_number(y), fill="#222222", font_family="Helvetica",
                       font_size=_number(size), text_anchor=anchor, **attrs)
        node.text = content
        return node

    def px(x):
        return left + (x-options.xmin) / (options.xmax-options.xmin) * pw

    current_unit = primary

    def py(y):
        low, high = bounds[current_unit]
        return top + (high-y) / (high-low) * ph

    def coordinates(points):
        if arrays and len(points) > 8:  # The same strings, formatted from arrays.
            xy = np.array(points, dtype=float)
            return _path_data(px(xy[:, 0]), py(xy[:, 1]))
        return "M " + " L ".join(f"{_number(px(x))} {_number(py(y))}" for x, y in points)

    def path(points, **attrs):
        if points:
            return element("path", d=coordinates(points), **attrs)

    if options.grid:
        for i, segment in enumerate(grid_segments(options.xmin, options.xmax, ymin, ymax,
                                                  spacing, options.offset if len(traces) > 1 else 0.0, options.grid_interval)):
            path(segment, fill="none", stroke="#777777", stroke_width="0.5", stroke_opacity="0.18", id=f"grid-{i}")

    peak_boxes = []
    for index, (trace, parts, baseline) in enumerate(zip(traces, runs, baselines)):
        current_unit = trace.unit
        ymin, ymax = bounds[current_unit]
        color = to_hex(to_rgba(trace.color))
        stroke = dict(stroke=color, stroke_width=_number(options.linewidth), stroke_linecap="butt", stroke_linejoin="round")
        floor = baseline + fill_floor(trace.rt, trace.y)
        for run_index, part in enumerate(parts):
            identity = f"trace-{index}-{run_index}"
            whole_line = False
            if isinstance(part, list):
                points = [(x, y+baseline) for x, y in part]
            else:
                # Curve runs use the same float operations on arrays. Uncut curves
                # take the two common outcomes below; others use the point code.
                xs, ys = part[0], part[1] + baseline
                whole_line = len(xs) > 1 and bool(((ys >= ymin) & (ys <= ymax)).all()) and bool(np.isfinite(np.diff(xs)).all())
                if whole_line and options.fill and ymin <= floor <= ymax:
                    sx, sy = px(xs), py(ys)
                    length = float(np.hypot(*np.diff(np.column_stack((sx, sy)), axis=0).T).sum())
                    if length > 0:
                        floor_y = py(floor)
                        closure = abs(float(sy[-1])-floor_y) + abs(float(sx[-1])-float(sx[0])) + abs(float(sy[0])-floor_y)
                        element("path", d=_path_data(np.r_[sx, sx[-1], sx[0]], np.r_[sy, floor_y, floor_y]) + " Z", fill=color,
                                fill_opacity=_number(options.opacity), stroke_dasharray=f"{_number(length)} {_number(closure + 1)}",
                                id=identity, **stroke)
                        continue
                if whole_line and not options.fill:
                    element("path", d=_path_data(px(xs), py(ys)), fill="none", id=f"{identity}-line-0", **stroke)
                    continue
                points = list(zip(xs.tolist(), ys.tolist()))
            uncut_y = ymin <= floor <= ymax and all(ymin <= y <= ymax for _, y in points)
            if options.fill and len(points) > 1 and uncut_y:
                # One editable filled/stroked path. A single dash covers only the
                # measured curve; its gap hides the artificial return to baseline.
                polygon = [*points, (points[-1][0], floor), (points[0][0], floor)]
                screen = np.array([(px(x), py(y)) for x, y in points])
                length = float(np.hypot(*np.diff(screen, axis=0).T).sum())
                closure = abs(py(points[-1][1])-py(floor)) + abs(px(points[-1][0])-px(points[0][0])) + abs(py(points[0][1])-py(floor))
                if length > 0:
                    element("path", d=coordinates(polygon) + " Z", fill=color, fill_opacity=_number(options.opacity),
                            stroke_dasharray=f"{_number(length)} {_number(closure + 1)}", id=identity, **stroke)
                    continue
            if options.fill and len(points) > 1:
                polygon = _crop_polygon([(points[0][0], floor), *points, (points[-1][0], floor)], ymin, ymax)
                if polygon:
                    element("path", d=coordinates(polygon) + " Z", fill=color,
                            fill_opacity=_number(options.opacity), stroke="none", id=f"{identity}-fill")
            # _crop_line returns an in-range curve as its single segment.
            for segment_index, segment in enumerate([points] if whole_line else _crop_line(points, ymin, ymax)):
                path(segment, fill="none", id=f"{identity}-line-{segment_index}", **stroke)
            if len(points) == 1 and ymin <= points[0][1] <= ymax:
                element("circle", cx=_number(px(points[0][0])), cy=_number(py(points[0][1])),
                        r=_number(options.linewidth/2), fill=color, id=identity)
        if options.spectrum:
            shift = 0 if options.layout == 'overlay' else (len(traces) - 1 - index) * options.offset
            candidates = select_peak_labels(np.asarray(trace.rt) + shift, np.asarray(trace.y), options.xmin, options.xmax,
                                             centroid=trace.centroid, view_width_px=pw)
            for position, intensity in candidates:
                y = intensity + baseline
                if not ymin <= y <= ymax or py(y) < top + tf * 1.5:
                    continue
                label = f'{position-shift:g}' if trace.unit == 'mAU' else format_mz_label(position-shift)
                label_x = min(left+pw-len(label)*tf*.3, max(left+len(label)*tf*.3, px(position)))
                box = (label_x-len(label)*tf*.3-2, py(y)-tf*1.5, label_x+len(label)*tf*.3+2, py(y))
                if any(box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1] for other in peak_boxes):
                    continue
                peak_boxes.append(box)
                text(label_x, py(y)-tf*.4, label, size=tf, anchor='middle', data_peak='true')

    current_unit = primary
    ymin, ymax = bounds[primary]
    path([(options.xmin, ymax), (options.xmin, ymin), (options.xmax, ymin)],
         fill="none", stroke="#222222", stroke_width="0.7", id="axes")
    for value in xt:
        element("line", x1=_number(px(value)), x2=_number(px(value)), y1=_number(py(ymin)),
                y2=_number(py(ymin)+4), stroke="#222222", stroke_width="0.7")
        text(px(value), py(ymin)+tf+7, _number(value), size=tf, anchor="middle", data_axis="x")
    for value in (yt if primary == 'counts' or not right_axis else []):
        element("line", x1=_number(left-4), x2=_number(left), y1=_number(py(value)),
                y2=_number(py(value)), stroke="#222222", stroke_width="0.7")
        text(left-8, py(value)+tf*.35, scientific(value), size=tf, anchor="end", data_axis="y")
    text(left+pw/2, height-5, options.x_label, anchor="middle")
    if primary == 'counts' or not right_axis:
        text(0, 0, 'Ion Counts' if primary == 'counts' else 'Absorbance (mAU)', anchor="middle", transform=f"translate({_number(font)} {_number(top+ph/2)}) rotate(-90)")
    if right_axis:
        current_unit = 'mAU'
        element('line', x1=_number(left+pw), x2=_number(left+pw), y1=_number(top), y2=_number(top+ph), stroke='#222222', stroke_width='0.7')
        for value in right_ticks:
            element('line', x1=_number(left+pw), x2=_number(left+pw+4), y1=_number(py(value)), y2=_number(py(value)), stroke='#222222', stroke_width='0.7')
            text(left+pw+8, py(value)+tf*.35, scientific(value), size=tf, data_axis='y-right')
        text(0, 0, 'Absorbance (mAU)', anchor='middle', transform=f'translate({_number(left+pw+right-font*.6)} {_number(top+ph/2)}) rotate(-90)')
    for index, (label, trace) in enumerate(zip(labels, traces)):
        x = left + pw + legend_gap + (right if right_axis else 0)
        y = top + legend_font * (1 + 1.55 * index)
        element("line", x1=_number(x), x2=_number(x+18), y1=_number(y-legend_font*.3),
                y2=_number(y-legend_font*.3), stroke=to_hex(to_rgba(trace.color)),
                stroke_width=_number(options.linewidth))
        text(x+24, y, label, size=legend_font, data_legend="true")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), legend_font, spacings, bounds


def export_chromatograms(path, traces, options):
    Path(path).write_bytes(render_chromatograms(traces, options))


def svg_is_vector(path):
    root = ET.fromstring(Path(path).read_bytes())
    tags = {node.tag.rsplit("}", 1)[-1] for node in root.iter()}
    return bool(tags & {"path", "line", "polyline"}) and "text" in tags and "image" not in tags
