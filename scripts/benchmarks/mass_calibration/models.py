"""Small calibration models with an explicit true-mass forward convention.

For ppm models, observed = true * (1 + error(rt, true) / 1e6).
The mass covariate is TRUE m/z, not observed m/z. Correction solves that
forward equation exactly on its positive, increasing branch.
"""
from dataclasses import dataclass

import numpy as np


MODEL_NAMES = ("constant_da", "constant_ppm", "rt_ppm", "mz_ppm", "rt_mz_ppm")
PARAMETERS = dict(zip(MODEL_NAMES, (1, 1, 2, 2, 3)))


from kmzconvert_core.mass_correction import Reference


@dataclass(frozen=True)
class Observation:
    reference: Reference
    observed_mz: float
    rt: float
    method: str

    def __post_init__(self):
        if not np.isfinite([self.observed_mz, self.rt]).all() or self.observed_mz <= 0 or self.rt < 0:
            raise ValueError("Invalid observation")

    @property
    def da_error(self):
        return self.observed_mz - self.reference.expected_mz

    @property
    def ppm_error(self):
        return 1e6 * self.da_error / self.reference.expected_mz


@dataclass(frozen=True)
class Model:
    name: str
    intercept: float
    rt_slope: float = 0.
    mz_slope: float = 0.
    rt_origin: float = 0.
    mz_origin: float = 0.
    condition: float = 1.
    training_count: int = 0

    def predict_error(self, rt, true_mz):
        """Return Da for constant_da; ppm for every other model."""
        return self.intercept + self.rt_slope * (np.asarray(rt) - self.rt_origin) + self.mz_slope * (np.asarray(true_mz) - self.mz_origin)

    def forward(self, true_mz, rt):
        true_mz = np.asarray(true_mz, dtype=float)
        error = self.predict_error(rt, true_mz)
        return true_mz + error if self.name == "constant_da" else true_mz * (1 + error * 1e-6)

    def correct(self, observed_mz, rt):
        """Invert on the positive increasing branch, without altering the input."""
        y = np.asarray(observed_mz, dtype=float)
        if not np.isfinite(y).all() or np.any(y <= 0) or not np.isfinite(rt):
            raise ValueError("Correction requires finite positive masses and finite scalar RT")
        if self.name == "constant_da":
            result = y - self.intercept
        else:
            alpha = 1 + (self.intercept + self.rt_slope * (rt - self.rt_origin) - self.mz_slope * self.mz_origin) * 1e-6
            beta = self.mz_slope * 1e-6
            discriminant = alpha * alpha + 4 * beta * y
            if alpha <= 0 or np.any(discriminant <= 0):
                raise ValueError("Calibration has no qualified increasing positive inverse")
            # Rationalized quadratic root avoids cancellation for small beta;
            # this also reduces exactly to y/alpha when beta is zero.
            result = 2 * y / (alpha + np.sqrt(discriminant))
            if np.any(alpha + 2 * beta * result <= 0):
                raise ValueError("Calibration inverse is not increasing")
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Correction produced invalid masses")
        return result


def fit(name, observations):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown calibration model: {name}")
    rows = list(observations)
    if len(rows) < PARAMETERS[name] or len({r.reference.id for r in rows}) != len(rows):
        raise ValueError("Too few or duplicate reference observations")
    errors = np.array([r.da_error if name == "constant_da" else r.ppm_error for r in rows])
    if name in ("constant_da", "constant_ppm"):
        return Model(name, float(np.median(errors)), training_count=len(rows))
    rt = np.array([r.rt for r in rows])
    mass = np.array([r.reference.expected_mz for r in rows])
    origins = [float(rt.mean()), float(mass.mean())]
    variables = []
    if name in ("rt_ppm", "rt_mz_ppm"):
        variables.append((rt, origins[0]))
    if name in ("mz_ppm", "rt_mz_ppm"):
        variables.append((mass, origins[1]))
    scales = [float(np.std(v)) for v, _ in variables]
    if any(s <= np.finfo(float).eps for s in scales):
        raise ValueError("Rank-deficient calibration design")
    design = np.column_stack([np.ones(len(rows))] + [(v-o)/s for (v, o), s in zip(variables, scales)])
    coefficients, _, rank, singular = np.linalg.lstsq(design, errors, rcond=None)
    condition = float(singular[0] / singular[-1])
    if rank != PARAMETERS[name] or condition > 1e8:
        raise ValueError("Rank-deficient or ill-conditioned calibration design")
    slopes = iter(float(c/s) for c, s in zip(coefficients[1:], scales))
    b = next(slopes) if name in ("rt_ppm", "rt_mz_ppm") else 0.
    c = next(slopes) if name in ("mz_ppm", "rt_mz_ppm") else 0.
    return Model(name, float(coefficients[0]), b, c, *origins, condition, len(rows))


def residual(model, observation):
    corrected = float(model.correct(observation.observed_mz, observation.rt))
    return 1e6 * (corrected - observation.reference.expected_mz) / observation.reference.expected_mz


def metrics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Metrics require finite residuals")
    return dict(mean_abs_ppm=float(np.mean(abs(values))), median_abs_ppm=float(np.median(abs(values))),
                rms_ppm=float(np.sqrt(np.mean(values**2))), worst_abs_ppm=float(np.max(abs(values))),
                bias_ppm=float(np.mean(values)))


def benchmark(observations):
    """All four held-out predictions; a failed fold invalidates aggregate CV."""
    rows = list(observations)
    results = []
    for name in MODEL_NAMES:
        record = dict(model=name, parameters=PARAMETERS[name], folds=[])
        try:
            model = fit(name, rows)
            record.update(full_model=model.__dict__, full_residuals_ppm=[residual(model, r) for r in rows])
            record["full_metrics"] = metrics(record["full_residuals_ppm"])
        except ValueError as error:
            record["full_failure"] = str(error)
        for i, row in enumerate(rows):
            fold = dict(held_out=row.reference.id, training_ids=[r.reference.id for j, r in enumerate(rows) if j != i])
            try:
                model = fit(name, [r for j, r in enumerate(rows) if j != i])
                fold.update(model=model.__dict__, predicted_error=float(model.predict_error(row.rt, row.reference.expected_mz)),
                            residual_ppm=residual(model, row), residual_degrees_of_freedom=len(rows)-1-PARAMETERS[name])
            except ValueError as error:
                fold["failure"] = str(error)
            record["folds"].append(fold)
        if all("residual_ppm" in f for f in record["folds"]):
            record["loo_metrics"] = metrics([f["residual_ppm"] for f in record["folds"]])
        else:
            record["loo_failure"] = "At least one fold cannot be fitted; partial folds are not ranked"
        results.append(record)
    return results


def correct_scan(model, mz, intensity, rt):
    """Preserve all samples, intensity object and RT; check the whole mass axis."""
    mz, intensity = np.asarray(mz), np.asarray(intensity)
    if mz.ndim != 1 or intensity.shape != mz.shape or not np.isfinite(intensity).all():
        raise ValueError("Invalid aligned scan arrays")
    if len(mz) > 1 and np.any(np.diff(mz) <= 0):
        raise ValueError("Source mass axis is not strictly increasing")
    corrected = model.correct(mz, rt)
    if len(corrected) > 1 and np.any(np.diff(corrected) <= 0):
        raise ValueError("Corrected mass axis is not strictly increasing")
    spacing = np.diff(corrected) / np.diff(mz)
    checks = dict(samples=len(mz), sample_count_unchanged=len(corrected) == len(mz),
                  intensity_unchanged=True, rt_unchanged=True, strictly_increasing=True,
                  minimum_spacing_ratio=float(spacing.min()) if len(spacing) else None,
                  maximum_spacing_ratio=float(spacing.max()) if len(spacing) else None)
    return corrected, intensity, checks
