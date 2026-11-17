"""Independent review regressions: compare against uncached font measurements."""
from dataclasses import replace

import matplotlib as mpl
import numpy as np
import pytest
import xml.etree.ElementTree as ET
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from kmzviewer import export_svg as svg


def trace(name):
    return svg.ExportTrace(name, 'TIC', 'red', np.array([0., 1.]), np.array([1., 2.]))


@pytest.fixture(autouse=True)
def clean_width_cache():
    svg._LABEL_WIDTHS.clear()
    yield
    svg._LABEL_WIDTHS.clear()


def test_cache_eviction_keeps_current_cached_labels():
    options = svg.ExportOptions()
    for i in range(4096):
        svg.legend_layout([trace(f'case-{i}.mzML')], options)
    assert len(svg._LABEL_WIDTHS) == 4096
    # The first label is a hit before eviction; the second triggers eviction.
    selected = [trace('case-0.mzML'), trace('new.mzML')]
    assert svg.render_chromatograms(selected, options).startswith(b'<?xml')


@pytest.mark.parametrize('changed', [{'font.weight': 'bold'}, {'font.style': 'italic'}])
def test_width_cache_respects_current_font_properties(changed):
    selected = [trace('warm.mzML')]
    options = svg.ExportOptions()
    svg.legend_layout(selected, options)
    with mpl.rc_context(changed):
        props = FontProperties(family=['Helvetica', 'Arial', 'DejaVu Sans'], size=1)
        expected = TextToPath().get_text_width_height_descent(svg.legend_label(selected[0]), props, False)[0]
        font, width, _ = svg.legend_layout(selected, options)
        assert width == expected * 1.05 * font + 24


@pytest.mark.parametrize('options,expected', [
    (dict(xmin=np.float32(0), xmax=np.float32(3)),
     'M 69.80000305 301.6 L 212.0000153 169.7481481 L 354.2000122 37.8962963 L 496.4000244 169.7481481'),
    (dict(xmin=0., xmax=3., spacing=np.float32(1.1)),
     'M 69.80000305 301.5999756 L 212.0000153 169.7481537 L 354.2000122 37.89630508 L 496.4000244 169.7481537'),
])
def test_numpy_scalar_options_preserve_original_coordinate_arithmetic(options, expected):
    # Expected strings independently captured from the staged point renderer.
    selected = [svg.ExportTrace('A.mzML', 'TIC', 'red', np.array([0., 1., 2., 3.]),
                                np.array([0., 1., 2., 1.]))]
    options = svg.ExportOptions(**options, fill=False, legend=False)
    root = ET.fromstring(svg.render_chromatograms(selected, options))
    line = next(n for n in root.iter() if n.get('id') == 'trace-0-0-line-0')
    assert line.get('d') == expected


def test_save_reuse_honours_new_destinations_cancel_retry_and_pending_changes(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from kmzviewer.gui.svg_preview import ExportDialog
    app = QApplication.instance() or QApplication([])
    t = trace('sample.mzML')
    dialog = ExportDialog([(1, t.filename, True)], (0., 1.), trace_data={1: t}, spectrum='ms1')
    targets = iter([str(tmp_path/'first'), str(tmp_path/'renamed.SVG'), '',
                    str(tmp_path/'missing'/'retry.svg'), str(tmp_path/'retry.svg'), str(tmp_path/'changed.svg')])
    defaults, errors = [], []
    def choose(*args):
        defaults.append(args[2])
        return next(targets), 'SVG (*.svg)'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', choose)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[2]))
    try:
        original = dialog.svg_bytes
        dialog.save_svg()
        assert (tmp_path/'first.svg').read_bytes() == original
        dialog.opacity.setValue(10)  # Fill is off; new state, identical SVG bytes.
        dialog.save_svg()  # Flush pending timer and save to a different filename.
        assert (tmp_path/'renamed.SVG').read_bytes() == original
        assert defaults[1] == str(tmp_path/'first.svg')
        saved = dialog.saved_path
        dialog.save_svg()  # Cancel.
        assert dialog.saved_path == saved
        dialog.save_svg()  # Missing directory raises OSError without closing dialog.
        assert errors and dialog.saved_path == saved
        dialog.save_svg()  # Retry same bytes, new destination.
        assert (tmp_path/'retry.svg').read_bytes() == original
        dialog.line_width.setValue(4)  # Real geometry/style change still debouncing.
        assert dialog._timer.isActive()
        dialog.save_svg()
        assert not dialog._timer.isActive()
        assert (tmp_path/'changed.svg').read_bytes() == dialog.svg_bytes != original
        assert dialog.svg_bytes == svg.render_chromatograms(dialog.selected_traces(), dialog.export_options())
    finally:
        dialog.close()
        app.processEvents()
