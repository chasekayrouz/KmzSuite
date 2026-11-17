"""Range/frame updates paint once; stacked rebuilds lay panes out once."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication

from kmzviewer.comparison import SpectrumFrame
from kmzviewer.gui.ms1_spectra import SpectrumPanel
from kmzviewer.gui.plots import ChromatogramPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class Paints(QObject):
    def __init__(self, widget):
        super().__init__()
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, _obj, event):  # noqa: N802
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def settle(qapp, panel):
    for _ in range(3):
        qapp.processEvents()
        panel.grab()
    qapp.processEvents()


def views(panel):
    return [plot.getViewBox() for plot in panel._plots()]


def rows(count=4, points=400):
    x = np.linspace(0., 10., points)
    return [(tid, f"Trace {tid}", "#0072b2", x, np.abs(np.sin(x * (tid + 1))) * 10 ** tid + 1.)
            for tid in range(count)]


@pytest.mark.parametrize("mode", ["stacked", "overlay"])
def test_chromatogram_range_changes_paint_once(qapp, mode):
    panel = ChromatogramPanel(normalize_overlay=True)
    panel.resize(900, 600)
    panel.show()
    try:
        panel.set_traces(rows())
        panel.set_mode(mode)
        settle(qapp, panel)
        paints = Paints(panel.view.viewport())
        for change in (lambda: panel.set_x_range(2., 6.),
                       lambda: panel._plots()[0].getViewBox().scaleBy(x=.9),
                       panel.reset_view):
            change()
            # Transforms are already applied: painting must not re-dirty the scene.
            assert not any(view._matrixNeedsUpdate for view in views(panel))
            before = paints.count
            for _ in range(3):
                qapp.processEvents()
            assert paints.count - before == 1
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_spectrum_frame_and_labels_paint_once(qapp):
    panel = SpectrumPanel()
    panel.resize(600, 700)
    panel.show()
    try:
        mz = np.linspace(100., 900., 4000)
        frames = [SpectrumFrame(fid, f"f{fid}", "#0072b2", f"f{fid}", mz,
                                np.abs(np.sin(mz / (7. + fid))) * 1000 * (fid + 1), False, None)
                  for fid in range(3)]
        panel.show_frames(frames, "three")
        settle(qapp, panel)
        paints = Paints(panel.view.viewport())
        for scale in (2., .5, 3.):
            panel.show_frames([SpectrumFrame(f.file_id, f.name, f.color, f.caption, f.mz, f.intensity * scale,
                                             False, None) for f in frames], "next")
            assert panel._label_items
            assert not any(view._matrixNeedsUpdate for view in views(panel))
            before = paints.count
            for _ in range(3):
                qapp.processEvents()
            assert paints.count - before == 1
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()



def test_unqualified_versions_leave_transforms_to_the_paint(qapp, monkeypatch, unqualified):
    chromatograms, spectra = ChromatogramPanel(), SpectrumPanel()
    prepared, scene = [], type(chromatograms.view.scene())
    original = scene.prepareForPaint
    monkeypatch.setattr(scene, "prepareForPaint", lambda self: prepared.append(self) or original(self))
    mz = np.linspace(100., 900., 400)
    try:
        for panel, size in ((chromatograms, (900, 600)), (spectra, (600, 700))):
            panel.resize(*size)
            panel.show()
        chromatograms.set_traces(rows())
        settle(qapp, chromatograms)
        for qualified, start, scale in ((True, 3., 2000.), (False, 2., 1000.)):
            if not qualified:
                unqualified("PySide6")
            prepared.clear()
            chromatograms.set_x_range(start, 6.)
            spectra.show_frames([SpectrumFrame(0, "f0", "#0072b2", "f0", mz, np.abs(np.sin(mz / 7.)) * scale,
                                               False, None)], "one")
            if qualified:
                assert set(prepared) == {chromatograms.view.scene(), spectra.view.scene()}
            else:  # pyqtgraph applies the new view transforms when it paints, as before.
                assert not prepared
            settle(qapp, chromatograms)
            settle(qapp, spectra)
    finally:
        for panel in (chromatograms, spectra):
            panel.close()
            panel.deleteLater()
        qapp.processEvents()


def test_stacked_rebuild_lays_out_once_and_links_to_first_pane(qapp):
    panel = ChromatogramPanel()
    panel.resize(900, 1200)
    panel.show()
    try:
        panel.set_traces(rows(2))
        settle(qapp, panel)
        resized = []
        original = panel._new_plot

        def new_plot(bottom):
            plot = original(bottom)
            plot.getViewBox().sigResized.connect(lambda view: resized.append(view))
            return plot
        panel._new_plot = new_plot
        panel.set_traces(rows(12))
        first, *rest = views(panel)
        # GraphicsLayout would otherwise re-lay out (and resize) every earlier pane per addition.
        assert resized.count(first) == 1
        assert all(view.linkedView(0) is first for view in rest)
        assert len({view.width() for view in views(panel)}) == 1
        assert all(view.viewRange()[0] == pytest.approx(first.viewRange()[0]) for view in rest)
        assert all(label.isVisible() and label.textItem.toPlainText() == label.title
                   for label in panel._inplot_labels.values())
        panel.set_x_range(3., 4.)
        assert all(view.viewRange()[0] == pytest.approx([3., 4.]) for view in views(panel))
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_stacked_gesture_refits_only_for_new_windows_or_inputs(qapp):
    panel = ChromatogramPanel()
    panel.resize(900, 1200)
    panel.show()
    try:
        data = rows(8)
        panel.set_traces(data)
        settle(qapp, panel)
        windows = []
        original = panel._fit_all_y
        panel._fit_all_y = lambda window=None: (windows.append(window), original(window))[1]
        panel._plots()[0].getViewBox().translateBy(x=.3)
        # The first pane, then followers reporting one (geometry-mapped) window.
        assert 1 <= len(windows) <= 2
        fitted = [view.viewRange()[1] for view in views(panel)]
        original(windows[-1])
        assert [view.viewRange()[1] for view in views(panel)] == fitted
        # New display values invalidate the memo even for an identical window.
        panel.set_traces([(tid, title, color, x, y * 2.) for tid, title, color, x, y in data])
        windows.clear()
        panel._plots()[3].getViewBox().translateBy(x=0.)
        panel._x_changed(panel._plots()[3].getViewBox(), tuple(panel._plots()[3].viewRange()[0]))
        assert windows
        assert views(panel)[-1].viewRange()[1][1] > fitted[-1][1]
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()
