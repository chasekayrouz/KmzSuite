from pathlib import Path

import numpy as np
import pytest

from kmzcomet.model import Feature, Measurement, Result, Sample, Settings
from kmzcomet.pca import feature_pca
from kmzcore.cancel import Cancelled


def matrix_result(rows, *, samples=None):
    """Generic height matrix; no sources are needed for PCA."""
    samples = samples if samples is not None else len(rows[0])
    sources = [Sample(Path(f'sample-{i}.mzML'), f'S{i:03}', '0'*64, 1,
                      units={'number of detector counts'}) for i in range(samples)]
    features = [Feature(f'F{i:06}', 150.+i*13.1, 1.+i*.2, 1, []) for i in range(len(rows))]
    columns = [[Measurement(f.id, s.id, False, 'not_acquired' if rows[i][j] is None else 'weak_signal',
                            height=rows[i][j]) for i, f in enumerate(features)] for j, s in enumerate(sources)]
    return Result(Settings(), sources, features, columns, 0.)


ROWS = [[90., 10., 0., 0., 0.], [9000., 1000., 0., 0., 0.], [0., 30., 60., 10., 0.],
        [0., 0., 20., 10., 70.], [1., 1., 1., 1., 1.], [10., 20., 30., 20., 20.],
        [1., 3., 5., 19., 7.], [20., 4., 5., 9., 8.]]


def test_profile_pca_matches_independent_covariance_eigendecomposition():
    result = matrix_result(ROWS); data = feature_pca(result)
    raw = np.array(ROWS)
    normalized = raw / raw.sum(axis=1, keepdims=True)
    centered = normalized - normalized.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)[::-1]
    assert data.rank == 4
    np.testing.assert_allclose(data.variance_ratio, eigenvalues[:4]/eigenvalues.sum())
    np.testing.assert_allclose(data.scores @ data.components, centered, atol=1e-14)
    np.testing.assert_allclose(data.components @ data.components.T, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(data.scores[0], data.scores[1], atol=1e-14)
    np.testing.assert_allclose([p.total for p in data.points], raw.sum(axis=1))
    assert data.points[1].total == 100 * data.points[0].total
    assert result.measurements[0][0].height == 90.
    assert not data.scores.flags.writeable and not data.components.flags.writeable


def test_row_scaling_and_sample_permutation_preserve_profile_distances():
    data = feature_pca(matrix_result(ROWS))
    raw = np.array(ROWS)
    changed = raw[:, [2, 0, 4, 1, 3]] * np.arange(1, len(ROWS)+1)[:, None]
    other = feature_pca(matrix_result(changed.tolist()))
    np.testing.assert_allclose(data.variance_ratio, other.variance_ratio, atol=1e-14)
    np.testing.assert_allclose(data.scores @ data.scores.T, other.scores @ other.scores.T, atol=1e-14)


def test_missing_is_omitted_zero_is_observed_and_weak_values_are_retained():
    data = feature_pca(matrix_result([[5., None, 1.], [0., 3., 1.], [0., 0., 0.], [.01, .02, .03]]))
    assert [p.row for p in data.points] == [1, 3]
    assert data.missing_rows == ('F000000',) and data.zero_rows == ('F000002',)
    assert data.feature_count == 4 and data.sample_count == 3
    assert data.points[1].total == pytest.approx(.06)
    assert data.rank == 1


@pytest.mark.parametrize('rows,rank', [([[1.], [20.]], 0), ([[1., 2., 3.]], 0),
    ([[1., 2., 3.], [10., 20., 30.], [1000., 2000., 3000.]], 0),
    ([[1., 0.], [0., 2.], [3., 4.]], 1),
    ([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [1., 1., 1.]], 2)])
def test_rank_deficient_profiles_have_no_fabricated_components(rows, rank):
    data = feature_pca(matrix_result(rows))
    assert data.rank == rank and data.scores.shape == (len(rows), rank)
    if rank:
        assert data.variance_ratio.sum() == pytest.approx(1)


@pytest.mark.parametrize('rows', [[], [[None, 1.], [0., 0.]]])
def test_no_eligible_rows_returns_explicit_empty_result(rows):
    data = feature_pca(matrix_result(rows, samples=2))
    assert data.scores.shape == (0, 0) and not data.points and data.rank == 0


@pytest.mark.parametrize('bad', [-1., float('inf'), float('nan')])
def test_invalid_heights_are_not_silently_imputed(bad):
    with pytest.raises(ValueError, match='finite, nonnegative'):
        feature_pca(matrix_result([[bad, 1.]]))


def test_small_units_normalize_without_a_pseudocount():
    first = feature_pca(matrix_result(ROWS))
    second = feature_pca(matrix_result((np.array(ROWS)*1e-250).tolist()))
    np.testing.assert_allclose(first.scores, second.scores, atol=1e-14)
    np.testing.assert_allclose(first.variance_ratio, second.variance_ratio, atol=1e-14)


def test_cancellation_after_svd_discards_results(monkeypatch):
    original = np.linalg.svd; completed = False
    def svd(*args, **kwargs):
        nonlocal completed
        value = original(*args, **kwargs); completed = True
        return value
    def checkpoint():
        if completed: raise Cancelled()
    monkeypatch.setattr(np.linalg, 'svd', svd)
    with pytest.raises(Cancelled):
        feature_pca(matrix_result(ROWS), checkpoint=checkpoint)
