"""Synthetic, offline qualification of packaged spectral exploration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from kmzviewer.reference_pack import ReferencePack
from kmzviewer.spectral_analysis import save_spectral_analysis
from kmzviewer.spectral_matching import prepare_search_peaks
from kmzviewer.spectral_motifs import search_motifs


def fixture_pack(query):
    """Two artificial spectra test direct and shifted evidence, not chemistry."""
    original = np.asarray(query["peaks"], dtype=np.float32)
    shift = 14.
    metadata, arrays, offsets = [], [], [0]
    for index, delta in enumerate((0., shift)):
        peaks = original.copy()
        peaks[:, 0] -= delta
        peaks = peaks[peaks[:, 0] > 0]
        metadata.append(dict(name=f"Synthetic reference {index + 1}", accession=f"SYNTHETIC{index + 1}",
            usi=f"mzspec:synthetic:reference:scan:{index + 1}", library="Synthetic qualification",
            precursor_mz=float(query["precursor_mz"] - delta), polarity=query["polarity"], charge=query.get("charge"),
            smiles="", inchi="", structure_key=f"synthetic:{index + 1}", adduct="", provenance="Artificial test spectrum",
            quality="", instrument="Synthetic", collision_energy=""))
        arrays.extend(peaks)
        offsets.append(len(arrays))
    return ReferencePack(metadata, np.asarray(arrays, dtype=np.float32), np.asarray(offsets, dtype=np.int64),
        dict(schema="synthetic-qualification", libraries=["Synthetic qualification"]))


def fixture_motifs(query, settings, cancel=None):
    # Exercise the bundled resource and real matcher before substituting a
    # generic pattern to guarantee that the evidence UI has a populated row.
    real = search_motifs(query, settings, cancel=cancel)
    assert real["provenance"]["database_records"] == 996
    maximum = max(y for _, y in query["peaks"])
    peaks = [[mz, y] for mz, y in query["peaks"]
             if mz < query["precursor_mz"] - 1 and y >= settings.relative_noise * maximum][:3]
    return dict(hits=[dict(name="Synthetic fragment pattern", motif_id="synthetic-1", motifset="Synthetic qualification",
        score=.8, matched_peaks=len(peaks), matched_fragments=len(peaks), matched_losses=0,
        motif_coverage=1., query_coverage=.8, fragment_peaks=peaks, loss_peaks=[],
        matches=[dict(type="fragment", reference_mz=mz, query_mz=mz, query_index=i, query_intensity=intensity,
                      error_da=0., query_difference_mz=None) for i, (mz, intensity) in enumerate(peaks)],
        source_url="", annotation_url="", interpretation="Artificial pattern for UI qualification only.")],
        provenance=dict(database_records=996, searched_records=real["provenance"]["searched_records"],
                        method="Synthetic display fixture; public resource integrity checked"),
        preparation=real["preparation"], warnings=[])


def _hdf5_roundtrip(output):
    """Exercise bundled h5py/HDF5, compressed input, conversion and SQLite search."""
    import h5py
    from kmzviewer.dreams_reference import DIMENSIONS, build_reference, search_reference

    with tempfile.TemporaryDirectory(prefix="spectral-hdf5-", dir=output) as folder:
        folder = Path(folder)
        source = folder / "synthetic-reference.hdf5"
        embedding = np.zeros((3, DIMENSIONS), dtype=np.float32)
        embedding[:, :2] = [[1., 0.], [.9, .1], [.6, .8]]
        exact_mass = 144.08012345
        spectra = np.array([[[100., exact_mass, 0.], [10., 20., 0.]],
                            [[120., 144., 0.], [5., 40., 0.]],
                            [[50., 150., 0.], [8., 30., 0.]]], dtype=np.float64)
        with h5py.File(source, "w") as file:
            file.create_dataset("DreaMS_embedding", data=embedding, compression="gzip")
            file.create_dataset("spectrum", data=spectra, compression="gzip")
            string = h5py.string_dtype("utf-8")
            for key, values in dict(
                    IDENTIFIER=["MassSpecGymID000001", "MassSpecGymID000002", "MassSpecGymID000003"],
                    INCHIKEY=["synthetic-same", "synthetic-same", "synthetic-other"],
                    name=["Synthetic reference one", "Synthetic repeat", "Synthetic reference two"],
                    smiles=["CC", "CC", "CCC"], adduct=["[M+H]+"] * 3).items():
                file.create_dataset(key, data=np.asarray(values, dtype=object), dtype=string)
            file["precursor_mz"] = [200., 200., 400.]
            file["COLLISION_ENERGY"] = [10., np.nan, 30.]
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        target = folder / "prepared"
        manifest = build_reference(source, target, expected_sha256=checksum, expected_count=3)
        assert manifest["count"] == 3, "Synthetic DreaMS conversion lost reference rows"
        result = search_reference(embedding[0], target)
        assert [h["accession"] for h in result["hits"]] == ["MassSpecGymID000001", "MassSpecGymID000003"], \
            "Packaged DreaMS search did not preserve independent ranking/grouping"
        assert abs(result["hits"][1]["scores"]["dreams"] - .6) < .0002, \
            "Packaged float16 DreaMS cosine exceeded its qualified numerical tolerance"
        assert result["references"]["MassSpecGymID000001"]["peaks"] == [[100., 10.], [exact_mass, 20.]], \
            "Packaged DreaMS conversion changed native reference masses or intensities"
    receipt = dict(ok=True, synthetic_only=True, reference_rows=3, unique_groups=2,
                   h5py=h5py.__version__, hdf5=h5py.version.hdf5_version,
                   gzip_input=True, native_mass_preserved=True, online_requests=0)
    (Path(output) / "spectral-native-stack.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def step(main, source, output, state):
    """Advance one timer tick; return True only after all worker cleanup ends."""
    if "window" not in state:
        state["native_stack"] = _hdf5_roundtrip(output)
        window = main._new_spectral_window()
        calls = []
        def loader(*args, **kwargs):
            calls.append("local reference pack")
            return fixture_pack(window.query)
        window.pack_loader = loader
        window.motif_searcher = fixture_motifs
        window.show()
        window.load_source(source, 300., 1., 1)
        state.update(window=window, calls=calls, phase=0)
        return False
    window = state["window"]
    if window.runtime.job is not None:
        return False
    if state["phase"] == 0:
        assert window.query and window.query["precursor_mz"] == 300., "Synthetic MS2 source was not loaded"
        assert state["calls"] == [], "Reviewing a synthetic scan unexpectedly started reference retrieval"
        count = prepare_search_peaks(window.query, window.settings())["retained_count"]
        assert count >= 2, "Synthetic smoke requires at least two fragments after precursor exclusion"
        # The shared runtime fixture has two fragments plus its precursor.
        # Explicitly qualify this small fixture rather than weakening defaults.
        window.minimum.setValue(min(3, count))
        state["expected_fragments"] = count
        state["expected_motif_features"] = len(fixture_motifs(window.query, window.settings())["hits"][0]["matches"])
        window.search_button.click()
        state["phase"] = 1
    elif state["phase"] == 1:
        assert window.analysis is not None, window.statusBar().currentMessage()
        assert window.matches.rowCount() >= 1, "Synthetic local references produced no qualifying results: " + window.summary.text()
        assert window.motifs.rowCount() == 1, "Synthetic motif evidence row is missing"
        assert window.pairs.rowCount() == state["expected_fragments"], "Synthetic matched-fragment evidence is incomplete"
        assert len(window.analysis["library"]["prepared_query"]) == state["expected_fragments"], \
            "Prepared query changed between preview and search"
        path = output / "synthetic.kmzspectra"
        save_spectral_analysis(path, window.analysis)
        calls = list(state["calls"])
        window.open_analysis(path)
        assert state["calls"] == calls, "Opening saved results unexpectedly accessed the reference pack"
        window.tabs.setCurrentIndex(1)
        assert window.pairs.rowCount() == state["expected_motif_features"], "Saved motif pairs were not restored"
        window.tabs.setCurrentIndex(0)
        window.grab().save(str(output / "spectral-synthetic.png"))
        window.close()
        state["phase"] = 2
    else:
        return window not in main._gnps_windows
    return False
