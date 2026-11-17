"""Headless walk through open, extract, walk, and SVG export."""

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from kmzviewer.export_svg import ExportOptions, svg_is_vector
from kmzviewer.gui.app import MainWindow
from Tests.Viewer.mzml_factory import write_mzml


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def _wait_for_caption(window, qapp, fragment, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if fragment in window.spectrum.caption.text() and "loading" not in window.spectrum.caption.text():
            return window.spectrum.caption.text()
        time.sleep(0.02)
    return window.spectrum.caption.text()


def test_second_chromatogram_keeps_its_own_y_scale(qapp, tmp_path):
    first = tmp_path / "A.mzML"
    second = tmp_path / "B.mzML"
    write_mzml(
        first,
        [
            {"mz": [100.0], "intensity": [1.0], "rt": 1.0, "tic": 1.0e6, "polarity": 1},
            {"mz": [100.0], "intensity": [1.0], "rt": 2.0, "tic": 2.0e6, "polarity": 1},
        ],
    )
    write_mzml(
        second,
        [
            {"mz": [100.0], "intensity": [1.0], "rt": 1.0, "tic": 5.0e6, "polarity": 1},
            {"mz": [100.0], "intensity": [1.0], "rt": 2.0, "tic": 8.0e6, "polarity": 1},
        ],
    )
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    qapp.processEvents()
    try:
        assert window.load_paths_sync([str(first), str(second)]) == []
        qapp.processEvents()
        panes = window.plots._plots()
        assert len(panes) == 2
        assert panes[1].viewRange()[1][1] > 1.0e6

        window.plots.set_x_range(0.5, 1.5)
        qapp.processEvents()
        assert panes[1].viewRange()[1][1] > 1.0e6

        window.match_action.setChecked(True)
        qapp.processEvents()
        assert panes[0].viewRange()[1][1] == pytest.approx(panes[1].viewRange()[1][1], rel=0.05)

        window.match_action.setChecked(False)
        qapp.processEvents()
        assert panes[1].viewRange()[1][1] > panes[0].viewRange()[1][1]
        assert panes[1].viewRange()[1][1] > 1.0e6
    finally:
        window.close()
        qapp.processEvents()


def test_open_extract_walk_and_export(qapp, tmp_path):
    path = tmp_path / "Sample_A.mzML"
    write_mzml(
        path,
        [
            {
                "id": "scan-a",
                "mz": [273.1559, 300.0],
                "intensity": [12.0, 1.0],
                "rt": 1.0,
                "tic": 40.0,
                "polarity": 1,
                "centroid": True,
            },
            {
                "id": "ms2",
                "ms_level": 2,
                "mz": [273.1559],
                "intensity": [99.0],
                "rt": 1.2,
                "tic": 99.0,
                "polarity": 1,
            },
            {
                "id": "scan-b",
                "mz": [273.1559],
                "intensity": [5.0],
                "rt": 2.0,
                "tic": 15.0,
                "polarity": 1,
                "centroid": True,
            },
        ],
    )
    window = MainWindow()
    try:
        errors = window.load_paths_sync([str(path)])
        assert errors == []
        assert window.tree.topLevelItemCount() == 1
        file_item = window.tree.topLevelItem(0)
        assert file_item.text(0) == "Sample_A.mzML"
        assert file_item.child(0).text(0) == "TIC MS1"
        loaded = next(iter(window.session.files.values()))
        assert loaded.scans.scan_id == ["scan-a", "scan-b"]

        window.extract_sync("273.1559", 10, "all", False, [loaded.file_id])
        labels = [loaded.traces[index].label for index in range(len(loaded.traces))]
        assert labels[0] == "TIC MS1"
        assert "273.1559" in labels[1]
        assert loaded.traces[1].y.tolist() == [12.0, 5.0]

        window.plots.set_mode("overlay")
        window.plots.set_mode("stacked")
        qapp.processEvents()
        panes = window.plots._plots()
        assert len(panes) == 2
        window.plots.set_x_range(0.5, 1.5)
        qapp.processEvents()
        first_x = panes[0].viewRange()[0]
        second_x = panes[1].viewRange()[0]
        assert tuple(second_x) == pytest.approx(tuple(first_x))
        tic_y = panes[0].viewRange()[1]
        # The rendered line reaches 27.5 at the cropped right boundary.
        assert tic_y == pytest.approx((26.875, 40.625))
        assert window.autoscale_action.isChecked()
        assert not window.match_action.isChecked()
        from PySide6.QtGui import QKeySequence

        assert QKeySequence("Ctrl+F") in window.full_action.shortcuts()
        window.full_action.trigger()
        qapp.processEvents()
        full_x = window.plots.current_x_range()
        assert full_x is not None and full_x[0] < 1.0 and full_x[1] > 2.0
        for plot in window.plots._plots():
            assert tuple(plot.viewRange()[0]) == pytest.approx(full_x)

        window.plots.set_x_range(0.5, 1.5)
        window.match_action.setChecked(True)
        qapp.processEvents()
        panes = window.plots._plots()
        assert tuple(panes[0].viewRange()[1]) == pytest.approx(tuple(panes[1].viewRange()[1]), abs=0.2)
        assert panes[0].viewRange()[1][1] > 40
        window.autoscale_action.setChecked(False)
        panes[0].setYRange(0, 25, padding=0)
        qapp.processEvents()
        assert tuple(panes[1].viewRange()[1]) == pytest.approx((0, 25), abs=0.3)
        window.plots.set_x_range(1.6, 2.2)
        assert tuple(panes[0].viewRange()[1]) == pytest.approx((0, 25), abs=0.3)
        window.match_action.setChecked(False)
        window.autoscale_action.setChecked(True)
        window.full_action.trigger()
        from PySide6.QtCore import Qt

        file_item = window.tree.topLevelItem(0)
        file_item.child(1).setCheckState(0, Qt.CheckState.Unchecked)
        qapp.processEvents()
        assert len(window.plots._plots()) == 1
        file_item.child(1).setCheckState(0, Qt.CheckState.Checked)
        qapp.processEvents()
        assert len(window.plots._plots()) == 2

        from kmzviewer.gui.dialogs import ExportDialog, ExtractDialog

        extract = ExtractDialog([(loaded.file_id, loaded.name)])
        extract.mz_edit.setText("250-300")
        assert not extract.ppm.isEnabled()
        assert "ppm is not applied" in extract.note.text()
        extract.mz_edit.setText("273.1559, 301.1412")
        assert extract.ppm.isEnabled()
        assert extract.merge.isEnabled()
        export = ExportDialog(
            [(loaded.traces[0].trace_id, "Sample_A.mzML — TIC MS1", True)],
            (0.5, 1.5),
            (1.0, 2.0),
            single_trace=True,
        )
        assert export.auto_y.isChecked()
        assert export.xmin.value() == pytest.approx(0.5)
        assert export.xmax.value() == pytest.approx(1.5)

        tic = loaded.traces[0]
        window.inspect_rt(tic.trace_id, 1.05)
        caption = _wait_for_caption(window, qapp, "scan-a")
        assert "MS1 centroid" in caption
        assert "scan-a" in caption
        window.show()
        window.spectrum.resize(900, 280)
        qapp.processEvents()
        window.spectrum._update_labels()
        label_text = [item.toPlainText() for item in window.spectrum._label_items]
        assert "273.1559" in label_text
        window.walk_by(1)
        caption = _wait_for_caption(window, qapp, "scan-b")
        assert "scan-b" in caption
        assert "MS2" not in caption
        window.walk_by(1)
        caption = _wait_for_caption(window, qapp, "scan-b")
        assert "scan-b" in caption

        destination = tmp_path / "out.svg"
        x0, x1 = window.plots.data_x_range()
        window.export_trace_ids(
            [trace.trace_id for trace in loaded.traces],
            str(destination),
            ExportOptions(
                layout="stacked",
                width_in=6.5,
                height_in=4.5,
                linewidth=1.1,
                fontsize=10,
                xmin=x0,
                xmax=x1,
                auto_y=True,
                ymin=None,
                ymax=None,
                filename_labels=True,
                trace_labels=True,
                transparent=False,
            ),
        )
        text = destination.read_text(encoding="utf-8")
        assert svg_is_vector(destination)
        assert "(TIC)" in text
        assert "273.1559" in text
        assert "<image" not in text.lower()
    finally:
        window.close()
        qapp.processEvents()
