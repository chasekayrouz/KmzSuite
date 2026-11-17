"""Independent review regression: cancellation before motif candidate allocation.

This intentionally fails on the submitted spectral delta; no production fix is
included in the review. A request may arrive while the bundled library loads.
"""
from threading import Event

import pytest

from kmzcore.cancel import Cancelled
from kmzviewer import spectral_motifs


def test_cancel_during_library_load_skips_candidate_construction(monkeypatch):
    library = spectral_motifs._load_library()
    cancelled = Event()
    candidate_calls = []
    # The base implementation constructs candidates inside each _align call.
    owner = "_candidates" if hasattr(spectral_motifs, "_candidates") else "_align"
    actual = getattr(spectral_motifs, owner)

    def load():
        cancelled.set()
        return library

    def candidates(*args, **kwargs):
        candidate_calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(spectral_motifs, "_load_library", load)
    monkeypatch.setattr(spectral_motifs, owner, candidates)
    query = dict(peaks=[[145.082, 1.]], precursor_mz=163.094,
                 charge=1, polarity=1)
    with pytest.raises(Cancelled):
        spectral_motifs.search_motifs(query, {}, cancel=cancelled.is_set)
    assert candidate_calls == [], "Cancelled motif searches must not construct the full candidate table"
