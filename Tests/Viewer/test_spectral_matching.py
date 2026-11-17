"""Reference matching boundaries, evidence, complete filtering and safe caches."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzviewer.reference_pack import (
    ReferencePack, ensure_reference_pack, load_reference_pack, pack_status, _read_mgf,
)
from kmzviewer.spectral_matching import (
    SpectralSettings, align_peaks, prepare_search_peaks, search_reference_pack,
)


def query(**changes):
    result = dict(peaks=[[100., 20.], [150., 50.], [200., 30.]], precursor_mz=300.,
                  charge=1, polarity=1, representation="centroid")
    result.update(changes)
    return result


def pack(rows):
    metadata, peaks, offsets = [], [], [0]
    for i, row in enumerate(rows):
        values = dict(name=f"Public reference {i}", accession=f"REFERENCE{i}", usi="", library="Synthetic public fixture",
                      precursor_mz=300., polarity=1, charge=1, smiles="", inchi="", structure_key=f"fixture:{i}",
                      adduct="[M+H]+", provenance="Test fixture", quality="", instrument="", collision_energy="")
        values.update({k: v for k, v in row.items() if k != "peaks"})
        metadata.append(values)
        a = np.asarray(row.get("peaks", query()["peaks"]), dtype=np.float32)
        peaks.extend(a)
        offsets.append(len(peaks))
    return ReferencePack(metadata, np.asarray(peaks, dtype=np.float32), np.array(offsets, dtype=np.int64), {"schema": "test"})


def test_direct_and_shifted_channels_recover_different_evidence_without_identification():
    p = pack([{}, dict(precursor_mz=250., peaks=[[50., 20.], [100., 50.], [150., 30.]])])
    result = search_reference_pack(query(), SpectralSettings(), p)
    assert result["methods"]["direct"][0]["accession"] == "REFERENCE0"
    shifted = next(h for h in result["methods"]["neutral_loss"] if h["accession"] == "REFERENCE1")
    assert shifted["scores"]["neutral_loss"] == pytest.approx(1., abs=1e-5)
    assert shifted["alignment"]["shifted_count"] == 3
    assert shifted["alignment"]["direct_count"] == 0
    assert shifted["alignment"]["query_coverage"] == pytest.approx(1.)
    assert "not identity" in result["provenance"]["score_interpretation"]
    json.dumps(result, allow_nan=False)


def test_complete_corpus_filter_finds_candidate_beyond_top_fifty():
    rows = [{} for _ in range(60)]
    rows.append(dict(peaks=[[100., 20.], [150., 50.], [200., 30.], [220., 10.]]))
    result = search_reference_pack(query(), SpectralSettings(required_fragments=(220.,)), pack(rows))
    assert [h["accession"] for h in result["methods"]["direct"]] == ["REFERENCE60"]
    assert result["provenance"]["filtered_reference_spectra"] == 1


def test_fragment_and_loss_conditions_are_conjunctive_and_require_actual_evidence():
    p = pack([{}, dict(precursor_mz=350.)])
    settings = SpectralSettings(required_fragments=(100., 150.), required_losses=(100.,))
    result = search_reference_pack(query(), settings, p)
    assert [h["accession"] for h in result["methods"]["direct"]] == ["REFERENCE0"]


def test_polarity_is_not_inferred_and_other_polarity_cannot_rank():
    p = pack([dict(polarity=-1), {}])
    result = search_reference_pack(query(), SpectralSettings(), p)
    assert result["provenance"]["reference_spectra"] == 1
    assert result["methods"]["direct"][0]["accession"] == "REFERENCE1"
    with pytest.raises(ValueError, match="polarity"):
        search_reference_pack(query(polarity=0), SpectralSettings(), p)


def test_multiple_charge_disables_difference_channels_and_preserves_high_mz_fragments():
    q = query(charge=2, peaks=[[100., 20.], [200., 50.], [400., 30.]])
    result = search_reference_pack(q, SpectralSettings(), pack([dict(charge=2, peaks=q["peaks"])]))
    assert result["methods"]["direct"]
    assert not result["methods"]["hybrid"] and not result["methods"]["neutral_loss"]
    assert result["prepared_query"][-1][0] == 400.
    with pytest.raises(ValueError, match="multiple charge"):
        search_reference_pack(q, SpectralSettings(required_losses=(100.,)), pack([{}]))


def test_unknown_charge_remains_unknown_and_assumption_is_reported():
    q = query(charge=None)
    result = search_reference_pack(q, SpectralSettings(), pack([{}]))
    assert q["charge"] is None
    assert any("Charge is unreported" in w for w in result["preparation"]["warnings"])


def test_zero_charge_is_unknown_without_mutating_source():
    q = query(charge=0)
    result = search_reference_pack(q, SpectralSettings(), pack([{}]))
    assert result["methods"]["hybrid"] and q["charge"] == 0
    assert any("Charge is unreported" in w for w in result["preparation"]["warnings"])


def test_duplicate_structure_grouping_keeps_best_qualifying_spectrum():
    p = pack([dict(structure_key="same", peaks=[[100., 20.], [150., 50.], [210., 30.]]),
              dict(structure_key="same"), dict(structure_key="other")])
    result = search_reference_pack(query(), SpectralSettings(min_peaks=2), p)
    assert [h["accession"] for h in result["methods"]["direct"]] == ["REFERENCE1", "REFERENCE2"]
    assert len(p.metadata) == 3


def test_search_preparation_is_explicit_and_does_not_mutate_profile_apices():
    q = query(representation="profile", peaks=[[100., 10.], [100.01, 20.], [150., 100.], [200., .01], [299., 100.]])
    before = json.dumps(q)
    prepared = prepare_search_peaks(q, SpectralSettings())
    assert json.dumps(q) == before
    assert prepared["retained_count"] == 2
    assert "merge neighboring peaks" in prepared["description"]
    assert "qualified centroid" in prepared["warnings"][0]
    assert prepared["peaks"][:, 1].sum() == pytest.approx(1.)


def test_alignment_never_counts_a_peak_twice():
    result = align_peaks([[100., .5], [100.01, .5]], [[100.005, 1.]], 300., 300., .02)
    assert len(result["pairs"]) == 1
    assert result["query_coverage"] == pytest.approx(.5)
    assert result["reference_coverage"] == pytest.approx(1.)


def test_cancellation_before_search_does_not_build_cache():
    p = pack([{}])
    with pytest.raises(Cancelled):
        search_reference_pack(query(), SpectralSettings(), p, cancel=lambda: True)
    assert not hasattr(p, "_search_cache")


@pytest.mark.parametrize("changes", [dict(fragment_da=0), dict(fragment_da=float("nan")), dict(relative_noise=.9),
                                    dict(min_peaks=0), dict(min_peaks=1.5), dict(required_fragments=(float("nan"),))])
def test_invalid_search_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(SpectralSettings(), **changes).validate()


def mgf_fixture(library):
    def block(mode, i):
        return (f"BEGIN IONS\nPEPMASS=300\nMSLEVEL=2\nIONMODE={mode}\nCHARGE=1\n"
                f"NAME=Public fixture\nSPECTRUMID={library}-{i}\n100 20\n150 50\n200 30\nEND IONS\n\n")
    return block("Positive", 1) + block("Negative", 2) + block("", 3)


def test_pack_requires_explicit_download_preserves_polarity_and_validates_integrity(tmp_path):
    calls = []
    def transport(url, destination, **kwargs):
        calls.append(url)
        Path(destination).write_text(mgf_fixture(Path(destination).stem))
        return dict(url=url, bytes=Path(destination).stat().st_size, sha256="fixture")
    with pytest.raises(FileNotFoundError):
        ensure_reference_pack(tmp_path, downloader=transport)
    assert not calls
    created = ensure_reference_pack(tmp_path, allow_download=True, downloader=transport)
    assert len(calls) == 6 and all(u.startswith("https://external.gnps2.org/gnpslibrary/") for u in calls)
    assert {m["polarity"] for m in created.metadata} == {-1, 1}
    assert len(created.metadata) == 12
    assert all(r["skipped"]["unknown_polarity"] == 1 for r in created.provenance["libraries"])
    assert pack_status(tmp_path)["available"]
    reopened = ensure_reference_pack(tmp_path, downloader=lambda *a, **kw: pytest.fail("unexpected download"))
    assert np.array_equal(reopened.peaks, created.peaks)
    with (created.path / "metadata.json.gz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        load_reference_pack(created.path)


def test_cancelled_download_does_not_leave_a_partial_pack(tmp_path):
    def cancelled(url, destination, **kwargs):
        Path(destination).write_text("partial public file")
        raise Cancelled()
    with pytest.raises(Cancelled):
        ensure_reference_pack(tmp_path, allow_download=True, downloader=cancelled)
    assert not list(tmp_path.iterdir())


def test_expansion_reuses_the_verified_base_without_redownloading_it(tmp_path):
    calls = []
    def transport(url, destination, **kwargs):
        calls.append(url)
        Path(destination).write_text(mgf_fixture(Path(destination).stem))
        return dict(url=url, bytes=Path(destination).stat().st_size, sha256="fixture")
    base = ensure_reference_pack(tmp_path, allow_download=True, downloader=transport)
    calls.clear()
    expanded = ensure_reference_pack(tmp_path, allow_download=True, expanded=True, downloader=transport)
    assert len(calls) == 1 and calls[0].endswith("/BMDMS-NP.mgf")
    assert expanded.metadata[:len(base.metadata)] == base.metadata
    assert np.array_equal(expanded.peaks[:len(base.peaks)], base.peaks)
    assert len(expanded.provenance["libraries"]) == 7


def test_mgf_reader_preserves_negative_charge_and_does_not_assume_ms_level(tmp_path):
    path = tmp_path / "test.mgf"
    path.write_text(mgf_fixture("fixture").replace("CHARGE=1", "CHARGE=1-"))
    rows, _, _ = _read_mgf(path, "fixture", "Fixture")
    assert all(r["charge"] == -1 for r in rows)
    path.write_text(mgf_fixture("fixture").replace("MSLEVEL=2\n", ""))
    rows, _, skipped = _read_mgf(path, "fixture", "Fixture")
    assert not rows and skipped["non_ms2"] == 2
