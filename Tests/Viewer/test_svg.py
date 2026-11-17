"""SVG export must be editable vectors, not a screenshot."""

import numpy as np
import pytest

from kmzviewer.export_svg import ExportOptions, ExportTrace, export_chromatograms, svg_is_vector


@pytest.mark.parametrize('mode', ['stacked', 'overlay'])
@pytest.mark.parametrize('crop_y', [False, True])
def test_signed_svg_fill_is_below_curve_with_offsets_gaps_and_cropping(mode, crop_y):
    import re
    import xml.etree.ElementTree as ET
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication
    from kmzviewer.export_svg import render_chromatograms, plot_geometry
    app = QApplication.instance() or QApplication([])
    x = np.arange(7, dtype=float)
    y = np.array([-8., -4., -4., np.nan, -4., -7., -8.])
    traces = [ExportTrace('A', 'EWC', '#0072b2', x, y, 'mAU'),
              ExportTrace('B', 'EWC', '#d55e00', x, y, 'mAU')]
    # Crop the native minimum away horizontally, and optionally clip the fill
    # boundary vertically. Stacked offset must move the curve and fill together.
    base = 11 if mode == 'stacked' else 0
    options = _options(layout=mode, xmin=1.1, xmax=5.9, spacing=11, legend=False,
                       auto_y=not crop_y, ymin=base-7, ymax=base-1)
    svg = render_chromatograms(traces, options)
    root = ET.fromstring(svg)
    axes = next(n for n in root if n.get('id') == 'axes')
    coordinates = np.array([float(n) for n in re.findall(r'[-+\d.e]+', axes.get('d'))]).reshape(-1, 2)
    left, top = coordinates[0]
    right, bottom = coordinates[-1]
    low, high = plot_geometry(traces, options)[3]['mAU']
    image = QImage(round(options.width_in*72), round(options.height_in*72), QImage.Format.Format_RGBA8888)
    image.fill(0)
    painter = QPainter(image)
    QSvgRenderer(QByteArray(svg)).render(painter)
    painter.end()
    def alpha(rt, value):
        px = round(left+(rt-options.xmin)/(options.xmax-options.xmin)*(right-left))
        py = round(top+(high-value)/(high-low)*(bottom-top))
        assert 0 <= px < image.width() and 0 <= py < image.height()
        return image.pixelColor(px, py).alpha()
    assert alpha(1.5, base-2) == 0
    assert alpha(1.5, base-6) > 0
    assert alpha(3, base-6) == 0  # Nonfinite gaps must remain unfilled.
    np.testing.assert_array_equal(traces[0].y, y)


def _options(**overrides) -> ExportOptions:
    base = dict(
        layout="stacked",
        width_in=6.0,
        height_in=4.0,
        linewidth=1.0,
        fontsize=9.0,
        xmin=0.5,
        xmax=2.0,
        auto_y=True,
        ymin=None,
        ymax=None,
        filename_labels=True,
        trace_labels=True,
        transparent=False,
    )
    base.update(overrides)
    return ExportOptions(**base)


def test_svg_contains_paths_and_text_and_no_image(tmp_path):
    destination = tmp_path / "traces.svg"
    traces = [
        ExportTrace(
            "Sample_A.mzML",
            "EIC 273.1559 ±10 ppm",
            "#1f4e79",
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 10.0, 4.0, 0.0]),
        ),
        ExportTrace(
            "Sample_B.mzML",
            "TIC MS1",
            "#c05621",
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 3.0, 8.0, 2.0]),
        ),
    ]
    export_chromatograms(destination, traces, _options())
    text = destination.read_text(encoding="utf-8")
    assert svg_is_vector(destination)
    assert "Sample_A (EIC 273.1559)" in text
    assert "ppm" not in text and ".mzML" not in text
    assert "EIC 273.1559" in text
    assert "Retention time" in text
    assert "<image" not in text.lower()


def test_overlay_and_transparent_svg_stay_vector(tmp_path):
    destination = tmp_path / "overlay.svg"
    trace = ExportTrace(
        "Sample_A.mzML",
        "EIC 250-300",
        "#2f6f4e",
        np.array([0.0, 1.0, 2.0]),
        np.array([5.0, 6.0, 7.0]),
    )
    export_chromatograms(
        destination,
        [trace],
        _options(layout="overlay", transparent=True, auto_y=False, ymin=0.0, ymax=10.0),
    )
    assert svg_is_vector(destination)
    assert "data:image" not in destination.read_text(encoding="utf-8").lower()


def test_flat_geometry_and_shared_stack():
    import xml.etree.ElementTree as ET
    from kmzviewer.export_svg import render_chromatograms, stack_geometry
    traces = [ExportTrace('A & B.mzML', 'TIC', '#0072b2', np.array([0., 1., 2.]), np.array([0., 10., 0.])),
              ExportTrace('C.mzML', 'EIC', '#d55e00', np.array([0., 1., 2.]), np.array([0., 4., 0.]))]
    options = _options(xmin=0, xmax=3, offset=.5)
    runs, baselines, spacing, _ = stack_geometry(traces, options)
    assert spacing == 11
    assert baselines == [11, 0]
    assert runs[0][0][0] == (.5, 0)
    assert runs[1][0][0] == (0, 0)
    root = ET.fromstring(render_chromatograms(traces, options))
    tags = {node.tag.rsplit('}', 1)[-1] for node in root.iter()}
    assert not tags & {'g', 'clipPath', 'mask', 'use', 'image', 'rect'}
    assert all('clip-path' not in node.attrib for node in root.iter())
    assert root.attrib['width'] == '6in'
    assert 'A & B (TIC)' in ''.join(root.itertext())
    assert sum(node.text == 'Ion Counts' for node in root.iter()) == 1


def test_crop_interpolates_boundaries_without_altering_source():
    from kmzviewer.export_svg import visible_runs, stack_geometry
    trace = ExportTrace('A', 'TIC', '#000000', np.array([0., 2., 3., 4., 5.]), np.array([0., 10., np.nan, 8., 0.]))
    original = trace.y.copy()
    assert visible_runs(trace, .5, 1.5) == [[(.5, 2.5), (1.5, 7.5)]]
    assert len(visible_runs(trace, 0, 5)) == 2
    assert stack_geometry([trace], _options(xmin=.5, xmax=1.5))[2] == 8.25
    np.testing.assert_array_equal(trace.y, original)


def test_tick_interval_start_and_limits():
    import pytest
    from kmzviewer.export_svg import ticks
    assert ticks(0, 10, 3, 1) == [1, 4, 7, 10]
    assert ticks(-4, 4, 2, 1) == [-3, -1, 1, 3]
    with pytest.raises(ValueError, match='Too many'):
        ticks(0, 100, .01)
    with pytest.raises(ValueError):
        ticks(0, 10, 0)


def test_manual_y_cuts_geometry_not_masks():
    import xml.etree.ElementTree as ET
    from kmzviewer.export_svg import render_chromatograms
    trace = ExportTrace('A', 'TIC', '#0072b2', np.array([0., 1., 2.]), np.array([-5., 20., -5.]))
    svg = render_chromatograms([trace], _options(xmin=0,xmax=2,auto_y=False,ymin=0,ymax=10,linewidth=3,opacity=.25,legend=False))
    root = ET.fromstring(svg)
    lines = [n for n in root if n.get('stroke') == '#0072b2']
    assert len(lines) == 2  # Above-range apex must not be joined across the top.
    assert all(n.get('stroke-width') == '3' for n in lines)
    assert any(n.get('fill-opacity') == '0.25' for n in root)
    assert b'clip' not in svg


def test_empty_zero_and_invalid_exports():
    import pytest
    from kmzviewer.export_svg import render_chromatograms, stack_geometry
    zero = ExportTrace('A','TIC','#000000',np.array([0.,1.]),np.zeros(2))
    assert stack_geometry([zero],_options())[2] == 0
    assert render_chromatograms([zero],_options())
    with pytest.raises(ValueError, match='at least one'):
        render_chromatograms([],_options())
    with pytest.raises(ValueError):
        render_chromatograms([zero],_options(offset=float('nan')))


def test_legend_scientific_ticks_and_global_styles():
    import xml.etree.ElementTree as ET
    from kmzviewer.export_svg import render_chromatograms, legend_label
    traces=[ExportTrace('Example.SP.mzML','TIC MS1','#0072b2',np.array([0.,1.,2.]),np.array([2.,10.,3.])),
            ExportTrace('Example.mzML','EIC 273.1559 ±10 ppm','#d55e00',np.array([0.,1.,2.]),np.array([1.,8.,2.]))]
    assert [legend_label(t) for t in traces] == ['Example (TIC)','Example (EIC 273.1559)']
    options=_options(xmin=.25,xmax=2,linewidth=3,opacity=.25)
    root=ET.fromstring(render_chromatograms(traces,options))
    assert [n.text for n in root if n.get('data-axis') == 'x'] == ['1','2']
    assert all('e' in n.text for n in root if n.get('data-axis') == 'y')
    assert [n.text for n in root if n.get('data-legend')] == ['Example (TIC)','Example (EIC 273.1559)']
    paths=[n for n in root if n.get('id','').startswith('trace-')]
    assert len(paths) == 2  # Each cropped RT trace has one combined fill/stroke object.
    assert all(n.get('fill-opacity') == '0.25' and n.get('stroke-width') == '3' for n in paths)
    assert all(n.get('stroke-dasharray') for n in paths)
    options.legend=False
    root=ET.fromstring(render_chromatograms(traces,options))
    assert not any(n.get('data-legend') for n in root)
    assert 'Example' not in ''.join(root.itertext())


def test_grid_shear_matches_baselines_and_clips_at_plot_boundary():
    import pytest
    from kmzviewer.export_svg import grid_segments
    positive=grid_segments(0,5,0,30,10,1,1)
    # Origin 0 advances one minute per 10-count vertical stack step.
    assert ((0.,0),(3.,30)) in positive
    for a,b in positive:
        assert 0 <= a[0] <= 5 and 0 <= b[0] <= 5
        assert 0 <= a[1] <= 30 and 0 <= b[1] <= 30
        assert (b[0]-a[0]) == pytest.approx((b[1]-a[1])/10)
    negative=grid_segments(0,5,0,30,10,-1,1)
    assert ((3.,0),(0.,30)) in negative
    assert grid_segments(0,2,0,20,10,0,1) == [((0.,0),(0.,20)),((1.,0),(1.,20)),((2.,0),(2.,20))]
    with pytest.raises(ValueError,match='positive stack spacing'):
        grid_segments(0,2,0,20,0,1,1)


def test_combined_object_renders_like_separate_stroke_fill(tmp_path):
    import xml.etree.ElementTree as ET
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication
    from kmzviewer.export_svg import render_chromatograms
    app=QApplication.instance() or QApplication([])
    trace=ExportTrace('A','TIC','#0072b2',np.array([0.,.6,1.2,2.]),np.array([3.,10.,7.,2.]))
    svg=render_chromatograms([trace],_options(xmin=.2,xmax=1.8,legend=False,linewidth=2))
    root=ET.fromstring(svg)
    node=next(n for n in root if n.get('id','').startswith('trace-'))
    assert node.get('fill') == node.get('stroke')
    line=ET.Element(node.tag,dict(node.attrib))
    line.set('d',node.get('d').rsplit(' L ',2)[0])
    line.set('fill','none')
    line.attrib.pop('stroke-dasharray')
    node.set('stroke','none')
    root.insert(list(root).index(node)+1,line)
    separate=ET.tostring(root)
    def pixels(data):
        image=QImage(900,600,QImage.Format.Format_RGBA8888)
        image.fill(0)
        painter=QPainter(image)
        QSvgRenderer(QByteArray(data)).render(painter)
        painter.end()
        return np.frombuffer(image.bits(),dtype=np.uint8).copy()
    difference=np.abs(pixels(svg).astype(int)-pixels(separate).astype(int))
    assert np.count_nonzero(difference > 8) < 100


def test_right_legend_full_names_auto_fit_and_independent_font():
    import re
    import xml.etree.ElementTree as ET
    from dataclasses import replace
    from PySide6.QtGui import QFont, QFontMetricsF
    from PySide6.QtWidgets import QApplication
    from kmzviewer.export_svg import render_chromatograms, legend_label
    app = QApplication.instance() or QApplication([])
    trace = ExportTrace('Sample.mzML', 'TIC', '#0072b2', np.array([0., 1.]), np.array([0., 10.]))
    options = _options(xmin=0, xmax=1)
    def inspect(traces, settings):
        root = ET.fromstring(render_chromatograms(traces, settings))
        labels = [n for n in root if n.get('data-legend')]
        axes = next(n for n in root if n.get('id') == 'axes')
        plot_right = float(re.findall(r'[-+\d.e]+', axes.get('d'))[-2])
        assert [n.text for n in labels] == [legend_label(t) for t in traces]
        for node in labels:
            assert float(node.get('x')) > plot_right
            # Check actual Qt font metrics, independently of SVG layout metrics.
            font = QFont('Helvetica')
            font.setPixelSize(1000)
            rendered_width = QFontMetricsF(font).horizontalAdvance(node.text) * float(node.get('font-size')) / 1000
            assert float(node.get('x')) + rendered_width < settings.width_in * 72
            assert float(node.get('y')) < settings.height_in * 72
        return root, labels
    _, labels = inspect([trace], options)
    assert float(labels[0].get('font-size')) == 10
    long = replace(trace, filename='Long_sample_description_' * 7 + '.sp.mzML', label='EIC 273.1559 ±10 ppm')
    _, labels = inspect([long, trace], options)
    assert .1 <= float(labels[0].get('font-size')) < 10
    assert '…' not in labels[0].text
    root, labels = inspect([trace], replace(options, legend_fontsize=14, fontsize=11))
    assert labels[0].get('font-size') == '14'
    assert next(n for n in root if n.text == 'Ion Counts').get('font-size') == '11'
    _, labels = inspect([trace] * 30, options)
    assert float(labels[0].get('font-size')) < 10


def test_legend_manual_overflow_is_actionable():
    import pytest
    from kmzviewer.export_svg import render_chromatograms
    trace = ExportTrace('Sample.mzML', 'TIC', '#0072b2', np.array([0., 1.]), np.array([0., 10.]))
    with pytest.raises(ValueError, match='legend font size'):
        render_chromatograms([trace] * 30, _options(legend_fontsize=10))
    with pytest.raises(ValueError, match='font size'):
        render_chromatograms([trace], _options(legend_fontsize=float('nan')))
