"""Kmz2Viewer's pyteomics fill reads with the suite's pinned PSI-MS vocabulary, offline."""
import socket
from types import SimpleNamespace

import numpy as np

from kmz2viewer import index
from kmzcore import psi_ms
from Tests.Viewer.mzml_factory import write_mzml

NEWER_PARAM = ('<cvParam cvRef="MS" accession="MS:1004014" '
               'name="ion mobility isolation window target" value="1.2"/>')


def test_pyteomics_fill_uses_the_pinned_vocabulary_without_network(tmp_path, monkeypatch):
    path = tmp_path / "ms2.mzML"
    write_mzml(path, [dict(id="ms2", ms_level=2, mz=[100., 150.], intensity=[5., 9.], rt=1., tic=14., centroid=True)])
    # A valued term newer than psims' bundled copy: pyteomics needs the pinned release.
    path.write_text(path.read_text().replace("<scanList", NEWER_PARAM + "<scanList", 1))

    def offline(*args, **kwargs):
        raise AssertionError("network access")
    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(psi_ms, "_vocabulary", None)
    scan = SimpleNamespace(spectrum_index=0, tic=float("nan"), centroid=True)
    index._fill_with_pyteomics(path, [scan])
    np.testing.assert_array_equal(scan.mz, [100., 150.])
    np.testing.assert_array_equal(scan.intensity, [5., 9.])
    assert scan.tic == 14. and scan.n_peaks == 2 and psi_ms._vocabulary is not None
