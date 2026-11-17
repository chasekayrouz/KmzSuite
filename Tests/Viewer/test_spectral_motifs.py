"""Scientific evidence boundaries for local motif-overlap exploration."""
from copy import deepcopy
import json

import pytest

from kmzcore.cancel import Cancelled
from kmzviewer import spectral_motifs


def query(**changes):
    value = dict(precursor_mz=200., polarity=1, charge=1,
                 peaks=[[90., .001], [100., 100.], [120., 80.], [140., 60.], [200., 10.]],
                 preparation="Native centroid peaks")
    value.update(changes)
    return value


@pytest.fixture
def small_library(monkeypatch):
    def motif(polarity):
        return dict(motif_id="collision", motifset="Synthetic", annotation="Synthetic overlap",
                    analysis_polarity=polarity,
                    fragments=[[100., 1.], [100.005, .5], [120., .7]],
                    losses=[[60., .4], [100., 1.]])
    library = dict(source="Synthetic", source_url="https://example.org", source_commit="test",
                   source_sha256="test", retrieved="2026-11-13", license="Test fixture",
                   motifs=[motif("positive ionisation mode"), motif("negative ionisation mode")])
    monkeypatch.setattr(spectral_motifs, "_load_library", lambda: library)
    return library


def test_public_reference_recovers_indole_evidence_without_identity_input():
    # Public MassBank BML01185 / GNPS CCMSLIB00000205165, annotated L-tryptophan.
    # Retained >=1% MS2 peaks from its public GNPS reference; no name/structure is
    # passed to the matching function. This is not the user's private benchmark.
    peaks = [[103.0049, 22.], [115.0551, 23.], [118.0658, 177.], [130.0659, 24.],
             [132.0812, 123.], [142.0656, 30.], [144.0797, 268.], [145.0074, 50.],
             [146.0598, 1276.], [146.0938, 43.], [159.092, 145.], [170.059, 94.],
             [188.0695, 1741.], [188.1073, 36.], [188.1192, 24.], [188.1337, 22.]]
    original = query(precursor_mz=205.097, peaks=peaks, source_name="private filename")
    before = deepcopy(original)
    result = spectral_motifs.search_motifs(original, {})
    assert result["hits"][0]["name"] == "Indole substructure"
    assert result["hits"][0]["matched_peaks"] >= 8
    assert result["provenance"]["searched_records"] == 565
    assert result["provenance"]["database_records"] == 996
    assert original == before
    assert "private filename" not in json.dumps(result, allow_nan=False)


def test_one_query_ion_cannot_support_multiple_features_or_both_channels(small_library):
    result = spectral_motifs.search_motifs(query(), {})
    hit = result["hits"][0]
    assert hit["matched_peaks"] == 3
    assert hit["matched_fragments"] == 2 and hit["matched_losses"] == 1
    assert {m["query_index"] for m in hit["matches"]} == {1, 2, 3}
    assert hit["motif_coverage"] == pytest.approx(2.1 / 3.6)
    assert hit["query_coverage"] == pytest.approx(1.)
    assert result["preparation"]["original_count"] == 5
    assert result["preparation"]["retained_count"] == 3
    # Three motif features near the same measured ion cannot satisfy min_peaks.
    sparse = spectral_motifs.search_motifs(query(peaks=[[100., 100.]]), {})
    assert sparse["hits"] == []


def test_polarity_and_charge_are_never_silently_inferred(small_library):
    negative = spectral_motifs.search_motifs(query(polarity=-1, charge=-1), {})
    assert len(negative["hits"]) == 1
    assert negative["provenance"]["searched_records"] == 1
    unknown = spectral_motifs.search_motifs(query(polarity=0), {})
    assert unknown["hits"] == [] and "unknown" in unknown["warnings"][0]
    multiply = spectral_motifs.search_motifs(query(charge=2), dict(min_peaks=2))
    assert multiply["hits"][0]["matched_losses"] == 0
    assert "disabled" in multiply["warnings"][0]
    uncertain = spectral_motifs.search_motifs(query(charge=None), {})
    assert "assumes singly charged" in uncertain["warnings"][0]


def test_required_features_do_not_secretly_change_motif_scoring(small_library):
    baseline = spectral_motifs.search_motifs(query(), {})
    selected = spectral_motifs.search_motifs(query(), dict(required_fragments=(999.,), required_losses=(888.,)))
    assert baseline == selected


def test_cancel_before_and_during_search(small_library):
    with pytest.raises(Cancelled):
        spectral_motifs.search_motifs(query(), {}, cancel=lambda: True)
    calls = []
    def cancelled():
        calls.append(1)
        return len(calls) >= 3
    with pytest.raises(Cancelled):
        spectral_motifs.search_motifs(query(), {}, cancel=cancelled)


@pytest.mark.parametrize("changes,settings", [
    ({"precursor_mz": float("nan")}, {}), ({"polarity": 2}, {}),
    ({"charge": 1.5}, {}), ({"peaks": [[100., float("nan")]]}, {}),
    ({}, {"relative_noise": -1.}), ({}, {"fragment_da": 0.}), ({}, {"min_peaks": 2.5}),
])
def test_invalid_scientific_inputs_fail_closed(small_library, changes, settings):
    with pytest.raises(ValueError):
        spectral_motifs.search_motifs(query(**changes), settings)
