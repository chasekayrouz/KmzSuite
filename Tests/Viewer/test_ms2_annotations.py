"""Connector tiering keeps overlapping mass-difference labels apart."""

from kmz2viewer.annotate import place_annotations
from kmz2viewer.differences import PairAssignment
from kmz2viewer.formula import FormulaHit


def _pair(left, right, formula="H2O", precursor=False, height=10.0):
    hit = FormulaHit(formula=formula, theoretical=right - left, ppm_error=1.0, counts=(0, 2, 0, 1, 0))
    return PairAssignment(
        left_mz=left,
        right_mz=right,
        delta=right - left,
        left_intensity=height,
        right_intensity=height / 2,
        involves_precursor=precursor,
        formula_allowed=True,
        hits=[hit],
    )


def test_overlapping_connectors_use_different_tiers_and_outside_pairs_are_hidden():
    pairs = [
        _pair(100, 140, precursor=True, height=50),
        _pair(110, 150, height=40),
        _pair(400, 450, height=90),
    ]
    placed = place_annotations(pairs, 90, 160, view_width_px=600)
    assert [item.pair.left_mz for item in placed] == [100, 110]
    assert placed[0].tier != placed[1].tier
    assert place_annotations(pairs, 90, 160, view_width_px=20) == []
