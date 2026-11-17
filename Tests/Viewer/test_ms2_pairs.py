"""Top-peak selection and pairwise mass-difference assignments."""

import numpy as np

from kmz2viewer.differences import analyze_spectrum, select_anchors
from kmz2viewer.formula import MASS_H, MASS_O, FormulaIndex, FormulaSettings


def _peaks(pairs):
    mz = np.array([mass for mass, _height in pairs], dtype=np.float64)
    intensity = np.array([height for _mass, height in pairs], dtype=np.float64)
    return mz, intensity


def test_top_ten_plus_precursor_and_positive_unique_pairs():
    fragments = [(100 + index, 100 - index) for index in range(12)]
    mz, intensity = _peaks(fragments)
    anchors, added, virtual = select_anchors(mz, intensity, centroid=True, precursor_mz=250.0)
    assert len(anchors) == 11
    assert added and virtual
    assert sum(anchor.is_precursor for anchor in anchors) == 1
    heights = sorted((anchor.intensity for anchor in anchors if not anchor.virtual), reverse=True)
    assert heights == list(range(100, 90, -1))
    assert 100 + 11 not in [anchor.mz for anchor in anchors]

    analysis = analyze_spectrum(
        mz, intensity, centroid=True, precursor_mz=250.0, charge=1, formulas=FormulaIndex(), settings=FormulaSettings()
    )
    assert len(analysis.pairs) == 55
    assert all(pair.delta > 0 for pair in analysis.pairs)
    keys = {(pair.left_mz, pair.right_mz) for pair in analysis.pairs}
    assert len(keys) == 55
    assert any(pair.hits for pair in analysis.pairs if pair.involves_precursor)


def test_observed_precursor_is_not_duplicated():
    mz, intensity = _peaks([(200.0, 50.0), (150.0, 40.0), (120.0, 30.0)])
    anchors, added, virtual = select_anchors(mz, intensity, centroid=True, precursor_mz=200.0)
    assert not added and not virtual
    assert sum(anchor.is_precursor for anchor in anchors) == 1
    assert [anchor.mz for anchor in anchors].count(200.0) == 1


def test_charge_above_one_skips_precursor_losses_but_keeps_runners():
    water = 2 * MASS_H + MASS_O
    mz, intensity = _peaks([(300.0, 80.0), (300.0 - water, 70.0), (100.0, 10.0)])
    formulas = FormulaIndex()
    settings = FormulaSettings()
    charged = analyze_spectrum(
        mz, intensity, centroid=True, precursor_mz=300.0, charge=2, formulas=formulas, settings=settings
    )
    precursor_pairs = [pair for pair in charged.pairs if pair.involves_precursor]
    fragment_pairs = [pair for pair in charged.pairs if not pair.involves_precursor]
    assert precursor_pairs
    assert all(not pair.formula_allowed and pair.hits == [] for pair in precursor_pairs)
    assert fragment_pairs
    single = analyze_spectrum(
        mz, intensity, centroid=True, precursor_mz=300.0, charge=1, formulas=formulas, settings=settings
    )
    water_pairs = [
        pair for pair in single.pairs if pair.involves_precursor and pair.best and pair.best.formula == "H2O"
    ]
    assert water_pairs
    assert water_pairs[0].hits  # runner-up list is retained even when one formula wins
