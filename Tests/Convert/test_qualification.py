"""Keep legacy TIC extent accounting exact and tied to empty raw slots."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/validation'))
import qualify_references


@pytest.mark.parametrize('case, expected', [
    ('current_extent', True), ('full_index', True), ('extra_nonzero', False),
    ('missing_signal', False), ('wrong_time', False), ('wrong_level', False),
])
def test_reference_tic_extent_preserves_all_raw_signal(monkeypatch, case, expected):
    full = {'time': np.array([0., 0., 1., 1.]),
            'intensity': np.array([0., 0., 10., 0.]),
            'ms level': np.array([1, 2, 1, 2])}
    actual = {k: v[:3] for k, v in full.items()}
    reference = {k: v.copy() for k, v in full.items()}
    if case == 'current_extent':
        reference = actual
    elif case == 'extra_nonzero':
        reference['intensity'][-1] = 1.
    elif case == 'missing_signal':
        reference['intensity'][2] = 0.
    elif case == 'wrong_time':
        reference['time'][-1] += 1e-12
    elif case == 'wrong_level':
        reference['ms level'][-1] = 1
    reader = SimpleNamespace(
        index=[(0, 0, 0., 6, 0.), (0, 0, 0., 6, 0.),
               (0, 11, 60000., 6, 10.), (11, 0, 60000., 6, 0.)],
        scans=[SimpleNamespace(slot=2)],
        experiments=[SimpleNamespace(ms_level=1), SimpleNamespace(ms_level=2)],
        tic_arrays=lambda: tuple(actual.values()))
    monkeypatch.setattr(qualify_references, 'WiffReader', lambda _: nullcontext(reader))
    monkeypatch.setattr(qualify_references, 'iter_records', lambda path, tag: iter([
        {'id': 'TIC', 'arrays': reference if path == 'reference' else actual}]))
    result = qualify_references.audit_tic('reference', 'actual', 'raw')
    assert result['raw_index_contract_passed'] == expected
    if case == 'full_index':
        assert not result['exact'] and result['shared_prefix_exact']
        assert result['extra_reference_slots'] == 1
