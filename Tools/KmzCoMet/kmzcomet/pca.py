"""Exploratory PCA of feature profiles, independent of sample annotations or sources."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAPoint:
    row: int
    feature_id: str
    mz: float
    rt: float
    polarity: int
    total: float


@dataclass(frozen=True)
class FeaturePCA:
    points: tuple[PCAPoint, ...]
    scores: np.ndarray
    components: np.ndarray
    variance_ratio: np.ndarray
    missing_rows: tuple[str, ...]
    zero_rows: tuple[str, ...]
    sample_count: int
    feature_count: int
    units: str

    @property
    def rank(self):
        return len(self.variance_ratio)


def feature_pca(result, *, progress=lambda message: None, checkpoint=lambda: None):
    """Row-sum normalize, column-center, then exact thin SVD. Never impute.

    Rows are observations, samples are variables. No log transformation, sample
    scaling, unit-variance scaling, grouping or evidence threshold is applied.
    Native row sums are kept separately for color; the result is never modified.
    """
    checkpoint()
    count, samples = len(result.features), len(result.samples)
    if samples == 0:
        raise ValueError('Feature PCA requires at least one sample.')
    if len(result.measurements) != samples or any(len(c) != count for c in result.measurements):
        raise ValueError('Feature PCA requires a complete feature/sample matrix.')
    progress('Preparing normalized feature profiles…')
    profiles = np.empty((count, samples), dtype=np.float64)
    points, missing, zero = [], [], []
    for row, feature in enumerate(result.features):
        if row % 128 == 0:
            checkpoint()
        values = [column[row].height for column in result.measurements]
        if any(v is None for v in values):
            missing.append(feature.id)
            continue
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f'{feature.id}: PCA requires finite, nonnegative heights.')
        largest = float(values.max())
        if largest == 0:
            zero.append(feature.id)
            continue
        # Scale first to keep normalization stable for very small/large units.
        scaled = values / largest
        denominator = float(scaled.sum())
        total = largest * denominator
        if not np.isfinite(total):
            raise ValueError(f'{feature.id}: summed height exceeds the numeric range.')
        profiles[len(points)] = scaled / denominator
        points.append(PCAPoint(row, feature.id, feature.mz, feature.rt, feature.polarity, total))
    profiles = profiles[:len(points)]
    scores = np.empty((len(points), 0))
    components = np.empty((0, samples))
    ratios = np.empty(0)
    checkpoint()
    if len(points) > 1 and samples > 1:
        progress(f'Calculating PCA for {len(points):,} features × {samples:,} samples…')
        # Use the original profile scale for rank tolerance: proportional rows
        # can otherwise turn centering roundoff into a fictitious component.
        tolerance = np.finfo(float).eps * max(profiles.shape) * np.linalg.norm(profiles)
        profiles -= profiles.mean(axis=0)
        u, singular, vh = np.linalg.svd(profiles, full_matrices=False)
        checkpoint()  # LAPACK itself cannot be interrupted safely.
        rank = min(int(np.count_nonzero(singular > tolerance)), len(points)-1, samples-1)
        if rank:
            components = vh[:rank].copy()
            signs = np.sign(components[np.arange(rank), np.abs(components).argmax(axis=1)])
            components *= signs[:, None]
            scores = (u[:, :rank] * singular[:rank]) * signs
            variance = singular ** 2
            ratios = variance[:rank] / variance.sum()
    for array in (scores, components, ratios):
        array.setflags(write=False)
    units = set().union(*(sample.units for sample in result.samples))
    unit_label = next(iter(units)) if len(units) == 1 else 'mixed / unspecified native units'
    checkpoint()
    return FeaturePCA(tuple(points), scores, components, ratios, tuple(missing), tuple(zero),
                      samples, count, unit_label)
