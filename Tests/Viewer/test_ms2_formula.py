"""CHNOS formula table and Δmass lookup."""

import numpy as np

from kmz2viewer.formula import (
    MASS_C,
    MASS_H,
    MASS_N,
    MASS_O,
    FormulaIndex,
    FormulaSettings,
    build_formula_table,
    dbe_value,
    format_formula,
)


def test_known_formulas_and_formatting():
    assert format_formula((0, 2, 0, 1, 0)) == "H2O"
    assert format_formula((0, 3, 1, 0, 0)) == "NH3"
    assert format_formula((1, 0, 0, 1, 0)) == "CO"
    assert format_formula((1, 0, 0, 2, 0)) == "CO2"
    assert format_formula((6, 10, 0, 3, 0)) == "C6H10O3"
    assert dbe_value(0, 2, 0, 1, 0) == 0
    assert dbe_value(0, 3, 1) == 0


def test_default_table_keeps_small_neutral_losses_and_rejects_invalid_dbe():
    table = build_formula_table(FormulaSettings())
    formulas = {format_formula(tuple(int(v) for v in row)) for row in table.counts}
    assert "H2O" in formulas
    assert "NH3" in formulas
    assert "CO" in formulas
    assert "CO2" in formulas
    assert "" not in formulas
    assert np.all(table.dbe >= 0)
    assert np.all(table.dbe < 15)
    # CH has a non-integer DBE and is absent. C20H2 has DBE 20.
    assert "CH" not in formulas
    assert "C20H2" not in formulas
    assert "H4" not in formulas


def test_ppm_match_signed_error_and_best_candidate():
    index = FormulaIndex()
    settings = FormulaSettings()
    water = 2 * MASS_H + MASS_O
    exact = index.lookup(water, settings)
    assert exact[0].formula == "H2O"
    assert exact[0].ppm_error == 0 or abs(exact[0].ppm_error) < 1e-6

    observed = water * (1 + 10e-6)
    close = index.lookup(observed, settings)
    assert close[0].formula == "H2O"
    assert close[0].ppm_error == (observed - water) / water * 1e6
    assert close[0].ppm_error > 0

    outside = index.lookup(water * (1 + 100e-6), FormulaSettings(ppm=50))
    assert all(hit.formula != "H2O" for hit in outside)

    # Two theoretical masses around the observation; the nearer one wins and the other is kept.
    narrow = FormulaSettings(ppm=50, c_max=0, h_max=4, n_max=0, o_max=2, s_max=0, dbe_min=0, dbe_max=15)
    hits = index.lookup(water * (1 + 5e-6), narrow, limit=8)
    assert hits
    assert abs(hits[0].ppm_error) <= abs(hits[-1].ppm_error)
    assert len(hits) >= 1


def test_element_limits_are_respected():
    table = build_formula_table(FormulaSettings(c_max=0, h_max=4, n_max=1, o_max=1, s_max=0))
    assert table.counts[:, 0].max() == 0
    assert "C6H10O3" not in {format_formula(tuple(int(v) for v in row)) for row in table.counts}
