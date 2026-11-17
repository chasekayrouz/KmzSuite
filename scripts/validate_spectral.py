#!/usr/bin/env python3
"""Qualify spectral exploration locally, with optional fixed PUBLIC live control.

Default execution uses synthetic references and never contacts a service. A
private --query stays local even with --allow-download, which only downloads
fixed public reference collections. --public-control is the only live spectrum
retrieval path and accepts no private query. No DreaMS submission is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PySide6.QtWidgets import QApplication

from kmzcore.theme import apply_theme
from kmzcore.version import __version__
from kmzviewer.gnps import load_analysis
from kmzviewer.gnps_client import GnpsClient
from kmzviewer.gui.spectral import SpectralWindow
from kmzviewer.gui.spectral_smoke import fixture_motifs, fixture_pack
from kmzviewer.reference_pack import ensure_reference_pack, load_reference_pack
from kmzviewer.spectral_analysis import (
    load_spectral_analysis, new_spectral_analysis, save_spectral_analysis,
)
from kmzviewer.spectral_matching import SpectralSettings, search_reference_pack
from kmzviewer.spectral_motifs import search_motifs


def synthetic_query():
    return dict(peaks=[[100., 10.], [150., 50.], [180., 30.], [220., 20.]],
        precursor_mz=300., charge=1, polarity=1, representation="centroid",
        preparation="Synthetic centroid spectrum", precursor_source="synthetic selected ion",
        source_name="Synthetic qualification", scan_id="synthetic=1", spectrum_index=0,
        rt_min=1., collision_energy=None, target_mz=None, target_rt=None)


def public_query():
    hit = dict(usi="mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000001647",
        accession="CCMSLIB00000001647", name="Enterocin", adduct="M+Na")
    reference = GnpsClient().reference(hit)
    query = dict(peaks=reference["peaks"], precursor_mz=reference["precursor_mz"],
        charge=reference["charge"], polarity=1, representation="centroid",
        preparation="Fixed public GNPS reference spectrum", precursor_source="public spectrum resolver",
        source_name="Public GNPS reference", scan_id=hit["accession"], spectrum_index=0,
        rt_min=None, collision_energy=None, target_mz=None, target_rt=None)
    return query, hit["accession"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--query", type=Path, help="Saved .kmzgnps or .kmzspectra query; processed locally only")
    source.add_argument("--public-control", action="store_true", help="Retrieve fixed public GNPS reference over HTTPS")
    parser.add_argument("--pack", type=Path, help="Existing base-v1/expanded-v1 directory or reference-cache root")
    parser.add_argument("--allow-download", action="store_true", help="Allow fixed PUBLIC reference downloads only")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--fragment-da", type=float, default=.02)
    parser.add_argument("--relative-noise", type=float, default=.01)
    parser.add_argument("--output", type=Path, default=Path(".local/spectral-validation"))
    args = parser.parse_args()
    if (args.query or args.public_control) and not args.pack:
        parser.error("--query and --public-control require --pack")
    if args.allow_download and not args.pack:
        parser.error("--allow-download requires --pack")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.query and args.query.resolve() == (args.output / "analysis.kmzspectra").resolve():
        parser.error("Choose an output directory that does not overwrite the input snapshot")
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    app.setQuitOnLastWindowClosed(False)
    settings = SpectralSettings(fragment_da=args.fragment_da, relative_noise=args.relative_noise,
                                expanded=args.expanded).validate()
    public_accession = None
    if args.query:
        data = load_analysis(args.query) if args.query.suffix.lower() == ".kmzgnps" else load_spectral_analysis(args.query)
        query = data["query"]
    elif args.public_control:
        query, public_accession = public_query()
    else:
        query = synthetic_query()
    before = json.dumps(query, sort_keys=True)
    started = time.monotonic()
    if args.pack:
        if (args.pack / "manifest.json").is_file():
            pack = load_reference_pack(args.pack)
        else:
            pack = ensure_reference_pack(args.pack, allow_download=args.allow_download, expanded=args.expanded,
                progress=lambda fraction, message: print(message, flush=True))
        motifs = search_motifs(query, settings)
    else:
        pack = fixture_pack(query)
        motifs = fixture_motifs(query, settings)
    library = search_reference_pack(query, settings, pack)
    seconds = time.monotonic() - started
    assert json.dumps(query, sort_keys=True) == before, "The original query was mutated"
    if public_accession:
        self_hits = [h for h in library["methods"]["direct"] if h["accession"] == public_accession]
        assert self_hits and self_hits[0]["scores"]["direct"] >= .99, "Public reference self-match failed"
    data = new_spectral_analysis(query, settings, library, motifs)
    snapshot = args.output / "analysis.kmzspectra"
    save_spectral_analysis(snapshot, data)
    assert load_spectral_analysis(snapshot) == json.loads(json.dumps(data))
    def forbidden(*args, **kwargs):
        raise AssertionError("Reopening results must not start any search or download")
    window = SpectralWindow(cache_dir=args.output / "unused-cache", pack_loader=forbidden,
                            searcher=forbidden, motif_searcher=forbidden)
    try:
        window.open_analysis(snapshot)
        window.show()
        app.processEvents()
        assert window.runtime.job is None
        window.grab().save(str(args.output / "related-spectra.png"))
        if window.motifs.rowCount():
            window.tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(args.output / "motif-evidence.png"))
        report = dict(ok=True, version=__version__, private_query=bool(args.query),
            public_control=public_accession, seconds=round(seconds, 4), query_unmodified=True,
            reopened_without_search=True, query_peaks=len(query["peaks"]),
            prepared_query_peaks=len(library["prepared_query"]),
            reference_spectra=library["provenance"]["reference_spectra"],
            displayed={mode: len(hits) for mode, hits in library["methods"].items()},
            motif_matches=len(motifs["hits"]), public_reference_downloads_allowed=args.allow_download,
            private_spectrum_submissions=0)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        window.close()
        deadline = time.monotonic() + 10
        while not window.runtime.closed and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert window.runtime.closed, "Spectral window did not close safely"


if __name__ == "__main__":
    main()
