"""Offscreen MS2 browser: table, filters, spectrum, labels, and formula connectors."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from kmz2viewer.formula import MASS_H, MASS_O, FormulaSettings
from kmz2viewer.gui.app import MainWindow
from Tests.Viewer.mzml_factory import write_mzml

WATER = 2 * MASS_H + MASS_O


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def _library(path, other):
    write_mzml(
        path,
        [
            {
                "id": "scan-a",
                "ms_level": 2,
                "centroid": True,
                "rt": 5.2,
                "tic": 300.0,
                "precursor_mz": 300.0,
                "charge": 1,
                "precursor_intensity": 1.0e5,
                "collision_energy": 20.0,
                "mz": [300.0 - WATER, 180.0, 150.0, 120.0, 300.0],
                "intensity": [80.0, 40.0, 30.0, 20.0, 10.0],
            },
            {
                "id": "scan-b",
                "ms_level": 2,
                "centroid": True,
                "rt": 8.0,
                "tic": 10.0,
                "precursor_mz": 500.0,
                "mz": [111.0],
                "intensity": [9.0],
            },
        ],
    )
    write_mzml(
        other,
        [
            {
                "id": "scan-c",
                "ms_level": 2,
                "centroid": True,
                "rt": 5.4,
                "tic": 50.0,
                "precursor_mz": 300.002,
                "charge": 1,
                "mz": [132.0],
                "intensity": [12.0],
            }
        ],
    )


def test_table_filters_sorting_and_spectrum(qapp, tmp_path):
    first = tmp_path / "A.mzML"
    second = tmp_path / "B.mzML"
    _library(first, second)
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    qapp.processEvents()
    try:
        errors = window.load_paths_sync([str(first), str(second)])
        assert errors == []
        assert window.model.rowCount() == 3
        assert "3 matching" in window.count_label.text()

        window.precursor.setText("300")
        window._apply_filters()
        assert {window.model.scan_at(row).scan_id for row in range(window.model.rowCount())} == {"scan-a", "scan-c"}

        window.fragment.setText("132")
        window.fragment_tol.setValue(0.05)
        window.fragment_unit.setCurrentIndex(1)
        window._apply_filters()
        assert [window.model.scan_at(0).scan_id] == ["scan-c"]

        window.clear_filters()
        window.rt_min.setText("5")
        window.rt_max.setText("5.5")
        window._apply_filters()
        assert {window.model.scan_at(row).scan_id for row in range(window.model.rowCount())} == {"scan-a", "scan-c"}

        window.precursor.setText("300")
        window.fragment.setText(f"{300 - WATER:.6f}")
        window.fragment_tol.setValue(20)
        window.fragment_unit.setCurrentIndex(0)
        window._apply_filters()
        assert [window.model.scan_at(row).scan_id for row in range(window.model.rowCount())] == ["scan-a"]

        window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
        window.table.setCurrentIndex(window.proxy.index(0, 0))
        window.open_selected()
        qapp.processEvents()
        assert "scan-a" in window.spectrum.caption.text()
        assert window.spectrum.precursor_line is not None
        assert window.spectrum.precursor_line.isVisible()
        assert "300.0000" in window.spectrum.label_texts() or window.spectrum.label_texts()
        formulas = window.spectrum.formula_texts()
        assert "H2O" in formulas

        window.spectrum.plot.setXRange(100, 140, padding=0)
        qapp.processEvents()
        assert window.spectrum.label_texts()
        outside = [text for text in window.spectrum.formula_texts()]
        assert all(text for text in outside)

        window.spectrum.losses.setChecked(False)
        qapp.processEvents()
        assert window.spectrum.formula_texts() == []
        window.spectrum.reset_range()
        window.spectrum.losses.setChecked(True)
        qapp.processEvents()
        assert "H2O" in window.spectrum.formula_texts()

        window.settings = FormulaSettings(o_max=0)
        window._analysis_cache.clear()
        window._show_scan(window.current_scan(), keep_window=True)
        qapp.processEvents()
        assert "H2O" not in window.spectrum.formula_texts()

        window.clear_filters()
        window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
        window.table.setCurrentIndex(window.proxy.index(0, 0))
        window.open_selected()
        first_caption = window.spectrum.caption.text()
        window._step_spectrum(1)
        qapp.processEvents()
        assert window.spectrum.caption.text() != first_caption
        window.spectrum.previous_button.click()
        qapp.processEvents()
        assert window.spectrum.caption.text() == first_caption
    finally:
        window.close()
        qapp.processEvents()
