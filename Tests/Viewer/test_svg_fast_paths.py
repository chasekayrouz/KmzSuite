"""Array/native SVG paths must reproduce the point-by-point bytes exactly."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from kmzviewer import export_svg
from kmzviewer.export_svg import ExportOptions, ExportTrace, render_chromatograms


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


EDGES = [0., -0., 1., -1., .5, 1/3, 62., 438.123456789, 123456789.25, 123456789.35, 9999999999.4,
         9999999999.5, 1e10, 1e-4, 9.99999999995e-5, 1e-5, 5e-324, 1.7976931348623157e308,
         np.inf, -np.inf, np.nan, 999999999.95, 1.00000000005]


def test_path_data_formats_every_value_as_number(monkeypatch):
    rng = np.random.default_rng(3)
    values = np.r_[EDGES, -np.array(EDGES), rng.uniform(0, 2000, 4000), 10 ** rng.uniform(-7, 12, 4000),
                   rng.integers(0, 2**64, 4000, dtype=np.uint64).view(np.float64),
                   np.floor(rng.uniform(1e7, 1e8, 1000)) + (rng.integers(0, 4, 1000) * 2 + 1) / 8]
    values = values[:len(values) // 2 * 2]
    expected = "M " + " L ".join(f"{export_svg._number(x)} {export_svg._number(y)}"
                                 for x, y in zip(values[0::2], values[1::2]))
    assert export_svg._path_data(values[0::2], values[1::2]) == expected
    monkeypatch.setattr(export_svg, '_native_path_data', None)
    assert export_svg._path_data(values[0::2], values[1::2]) == expected


def test_native_path_data_is_used_and_rejects_partial_pairs():
    native = pytest.importorskip('kmzviewer._native')
    assert export_svg._native_path_data is native.path_data
    assert export_svg._native_path_data(np.empty(0)) == 'M '
    with pytest.raises(ValueError):
        export_svg._native_path_data(np.zeros(3))


def test_extremes_follow_builtin_min_max_including_signed_zero_and_nan():
    rng = np.random.default_rng(5)
    for _ in range(300):
        prefix = [float(v) for v in rng.choice([0., -0., 1., -1., 2.], rng.integers(0, 3))]
        if rng.random() < .3:
            prefix.append(int(rng.integers(-2, 3)))
        arrays = [rng.choice([0., -0., 1., -1., 3., np.nan, np.inf], rng.integers(0, 5)) for _ in range(rng.integers(0, 4))]
        flat = [*prefix, *(v for a in arrays for v in a.tolist())]
        low, high = export_svg._extremes(prefix, arrays)
        expected = (min(flat), max(flat)) if flat else (0., 0.)
        assert repr((low, high)) == repr(expected)
        assert [type(v) for v in (low, high)] == [type(v) for v in expected]


def _traces(rng, count):
    traces = []
    for i in range(count):
        n = int(rng.choice([1, 2, 5, 300]))
        x = np.sort(rng.uniform(-1, 11, n))
        y = rng.normal(0, 1, n) * 10.0 ** rng.integers(-2, 6)
        if rng.random() < .4:
            y[rng.random(n) < .1] = np.nan
        if rng.random() < .2:
            y[:] = np.where(rng.random(n) < .5, 0., -0.)
        traces.append(ExportTrace(f'S{i}.mzML', 'EIC', str(rng.choice(['#0072b2', 'red', '#D55E00'])), x, y,
                                  'mAU' if rng.random() < .3 else 'counts', bool(rng.random() < .15)))
    return traces


def test_array_runs_match_point_runs_and_formatters(monkeypatch):
    rng = np.random.default_rng(11)
    for case in range(120):
        traces = _traces(rng, int(rng.choice([1, 2, 4])))
        lo = float(rng.choice([0., -0., rng.uniform(-1, 5)]))
        options = ExportOptions(layout=str(rng.choice(['stacked', 'overlay'])), xmin=lo, xmax=lo + float(rng.uniform(.5, 12)),
                                fill=bool(rng.random() < .6), offset=float(rng.choice([0., .3, -0.])), legend=False,
                                grid=bool(rng.random() < .3), spectrum=bool(rng.random() < .3), y_tick_interval=float(10.0 ** rng.integers(-2, 6)))
        variants = [options]
        try:
            bounds = next(iter(export_svg.plot_geometry(traces, options)[3].values()))
            span = bounds[1] - bounds[0]
            variants.append(replace(options, auto_y=False, ymin=bounds[0] + span * .2, ymax=bounds[0] + span * .6))
        except ValueError:
            pass
        for variant in variants:
            outputs = []
            for mode in ('array', 'scalar', 'python'):
                with monkeypatch.context() as patch:
                    if mode == 'scalar':  # The original point geometry and coordinates.
                        patch.setattr(export_svg, '_python_numbers', lambda options: False)
                    if mode == 'python':
                        patch.setattr(export_svg, '_native_path_data', None)
                    try:
                        outputs.append(render_chromatograms(traces, variant))
                    except ValueError as exc:
                        outputs.append(str(exc))
            assert outputs[0] == outputs[1] == outputs[2], case


def test_public_runs_keep_bound_objects():
    trace = ExportTrace('A', 'TIC', '#000000', np.array([0., 2., 4.]), np.array([0., 10., 0.]))
    runs = export_svg.visible_runs(trace, 1, 3)
    assert runs == [[(1, 5.), (2., 10.), (3, 5.)]]
    assert [type(x) for x, _ in runs[0]] == [int, float, int]
    stick = replace(trace, centroid=True)
    assert export_svg.visible_runs(stick, 1, 3) == [[(2., 0.), (2., 10.)]]


def test_legend_widths_are_cached_measurements():
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    trace = ExportTrace('Cache_probe.mzML', 'TIC', '#0072b2', np.array([0., 1.]), np.array([0., 1.]))
    export_svg._LABEL_WIDTHS.clear()
    first = export_svg.legend_layout([trace], ExportOptions())
    label = export_svg.legend_label(trace)
    measured = TextToPath().get_text_width_height_descent(
        label, FontProperties(family=["Helvetica", "Arial", "DejaVu Sans"], size=1), False)[0]
    assert export_svg._LABEL_WIDTHS[export_svg.legend_font_properties()[1], label] == measured
    assert export_svg.legend_layout([trace], ExportOptions()) == first


def test_legend_width_cache_is_a_bounded_lru_and_keeps_current_widths(monkeypatch):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    metrics = TextToPath()
    properties = FontProperties(family=["Helvetica", "Arial", "DejaVu Sans"], size=1)
    traces = [ExportTrace(f'L{i:02d}{"W" * i}.mzML', 'TIC', '#0072b2', np.array([0., 1.]), np.array([0., 1.]))
              for i in range(20)]
    labels = lambda selected: [export_svg.legend_label(t) for t in selected]
    options = ExportOptions(width_in=40)
    monkeypatch.setattr(export_svg, '_LABEL_WIDTHS', type(export_svg._LABEL_WIDTHS)())
    monkeypatch.setattr(export_svg, '_LABEL_WIDTH_LIMIT', 8)
    # Warm, overfill in one call, then mix an evicted label with a cached one.
    for selected in (traces[:3], traces, [traces[2], traces[19], traces[3]]):
        expected = max(metrics.get_text_width_height_descent(label, properties, False)[0]
                       for label in labels(selected)) * 1.05
        font, width, _ = export_svg.legend_layout(selected, options)
        assert width == expected * font + 24 and len(export_svg._LABEL_WIDTHS) <= 8
    assert [label for _, label in export_svg._LABEL_WIDTHS] == labels(traces[14:19] + [traces[2], traces[19], traces[3]])


@pytest.mark.parametrize('kind,arrays', [(np.float32, False), (np.float64, False), (Fraction, False),
                                         (int, True), (float, True)])
@pytest.mark.parametrize('name', ['xmin', 'xmax', 'offset', 'spacing', 'width_in', 'fontsize', 'linewidth'])
def test_option_number_types_choose_paths_with_original_arithmetic(kind, arrays, name, monkeypatch):
    traces = [ExportTrace(f'S{i}.mzML', 'TIC', '#0072b2', np.linspace(0, 3, 40), np.sin(np.linspace(0, 3, 40)) * 7 + i)
              for i in range(2)]
    base = ExportOptions(xmin=0.25, xmax=2.75, offset=0.5, spacing=3.5, width_in=7.25, fontsize=12.0, linewidth=2.0)
    value = getattr(base, name)
    options = replace(base, **{name: kind(int(value) if kind is int else value)})
    assert export_svg._python_numbers(options) == arrays

    def render():
        try:
            return render_chromatograms(traces, options)
        except TypeError as exc:  # A Fraction offset fails in the original array shift.
            return repr(exc)
    actual = render()
    monkeypatch.setattr(export_svg, '_python_numbers', lambda options: False)
    assert actual == render()


@pytest.mark.parametrize('xmin,xmax,arrays', [(0, 2 ** 53, True), (1, 2 ** 53 + 1, False), (-2 ** 53 - 2, 0, False)])
def test_large_int_bounds_use_arrays_only_while_exact(xmin, xmax, arrays, monkeypatch):
    traces = [ExportTrace('S.mzML', 'TIC', '#0072b2', np.linspace(xmin, xmax, 40), np.cos(np.linspace(0, 3, 40)))]
    options = ExportOptions(xmin=xmin, xmax=xmax, x_tick_interval=2. ** 52, legend=False)
    assert export_svg._python_numbers(options) == arrays
    actual = render_chromatograms(traces, options)
    monkeypatch.setattr(export_svg, '_python_numbers', lambda options: False)
    assert actual == render_chromatograms(traces, options)


def test_preview_reuses_unchanged_render_and_identical_bytes(app, tmp_path, monkeypatch):
    from kmzviewer.gui import svg_preview
    data = {i: ExportTrace(f'S{i}.mzML', 'TIC', '#0072b2', np.linspace(0, 2, 50), np.sin(np.linspace(0, 3, 50)) + i)
            for i in range(3)}
    renders, loads = [], []
    original = svg_preview.render_with_layout
    monkeypatch.setattr(svg_preview, 'render_with_layout', lambda *args: (renders.append(1), original(*args))[1])
    dialog = svg_preview.ExportDialog([(i, f'S{i}', True) for i in data], (0, 2), trace_data=data, spectrum='ms1')
    try:
        load = dialog.preview.load
        dialog.preview.load = lambda content: (loads.append(1), load(content))[1]
        assert len(renders) == 1 and not dialog.shading.isChecked()
        target = tmp_path / 'saved.svg'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), 'SVG (*.svg)'))
        dialog.save_svg()
        assert len(renders) == 1 and target.read_bytes() == dialog.svg_bytes
        before = dialog.svg_bytes
        dialog.opacity.setValue(10)  # Shading is off: new inputs, identical bytes.
        dialog.update_preview()
        assert len(renders) == 2 and not loads and dialog.svg_bytes == before
        dialog.auto_spacing.setChecked(False)
        dialog.spacing.setText('0')
        dialog.update_preview()
        assert len(renders) == 3 and loads == [1] and b"{'counts': 0.0}" in dialog.svg_bytes
        dialog.spacing.setText('-0')  # Signed zero is a distinct input and description.
        dialog.update_preview()
        assert len(renders) == 4 and loads == [1, 1] and b"{'counts': -0.0}" in dialog.svg_bytes
        assert dialog.svg_bytes == render_chromatograms(dialog.selected_traces(), dialog.export_options())
        dialog.table.item(0, 0).setCheckState(svg_preview.Qt.CheckState.Unchecked)
        dialog.update_preview()
        assert len(renders) == 5 and len(loads) == 3 and dialog.selected_trace_ids() == [1, 2]
    finally:
        dialog.close()


def test_preview_rerenders_when_the_legend_font_context_changes(app, monkeypatch):
    import matplotlib as mpl
    from kmzviewer.gui import svg_preview
    data = {0: ExportTrace('Weighted_name.mzML', 'TIC', '#0072b2', np.linspace(0, 2, 50), np.linspace(0, 1, 50))}
    renders = []
    original = svg_preview.render_with_layout
    monkeypatch.setattr(svg_preview, 'render_with_layout', lambda *args: (renders.append(1), original(*args))[1])
    dialog = svg_preview.ExportDialog([(0, 'S0', True)], (0, 2), trace_data=data, spectrum='ms1')
    try:
        before = dialog.svg_bytes
        with mpl.rc_context({'font.weight': 'bold'}):
            dialog.update_preview()
            assert len(renders) == 2 and dialog.svg_bytes != before
            assert dialog.svg_bytes == render_chromatograms(dialog.selected_traces(), dialog.export_options())
        dialog.update_preview()
        assert len(renders) == 3 and dialog.svg_bytes == before
    finally:
        dialog.close()
