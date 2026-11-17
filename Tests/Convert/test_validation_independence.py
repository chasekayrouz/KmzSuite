"""Validators judge the converter's native output against the interpreted reference path.

A deliberately wrong native kernel (the fused spectrum kernels calibrate with a
slightly skewed slope) must make every validator reject the production output,
and the validators' reference side must run with the extension unusable.
Synthetic many-spectrum container only.
"""
import runpy
from pathlib import Path

import pytest

from kmzconvert_core import native, writer
from kmzconvert_core.reader import WiffReader
from Tests.Convert.test_parallel_writer import acquisition, correction  # noqa: F401 (fixture)

ROOT = Path(__file__).resolve().parents[2]
FUSED = ('profile_spectrum', 'profile_regions', 'encode_spectrum')


class Wrong:
    """The real extension, except that its fused kernels skew the calibration slope by 2**-30."""

    def __init__(self, real):
        self.real = real

    def __getattr__(self, name):
        kernel = getattr(self.real, name)
        if name not in FUSED:
            return kernel
        return lambda block, a, *rest: kernel(block, a * (1 + 2**-30), *rest)


class Unusable:
    def __getattr__(self, name):
        raise AssertionError(f'validator reference path used the native extension ({name})')


def production(path, root, label):
    """Full, sparse and corrected full/sparse outputs of the converter's reader."""
    mc = correction()
    with native.NativeWiffReader(path) as reader:
        return mc, {kind: writer.write_mzml(reader, root / f'{label}-{kind}.mzML', **options) for kind, options in (
            ('full', {}), ('sparse', dict(profile='sparse')), ('full-mc', dict(mass_correction=mc)),
            ('sparse-mc', dict(profile='sparse', mass_correction=mc)))}


def validators(path, root, mc, outputs, *, convert=True):
    """Run each validator's check of the given outputs; returns nothing or raises AssertionError."""
    from scripts import validate_mass_correction, validate_sparse_profile
    verify_output = runpy.run_path(str(ROOT / 'scripts/integration-parity.py'))['verify_output']
    if convert:
        # validate_sparse_profile's own conversion: production outputs, then the interpreted reference.
        _, reference, _, sparse = validate_sparse_profile.convert(path, root, 'converted')
        validate_sparse_profile.compare(reference, sparse)
    with WiffReader(path) as reader:
        reference = writer.write_mzml(reader, root / f'reference-{len(list(root.iterdir()))}.mzML')
    validate_sparse_profile.compare(reference, outputs['sparse'])
    with WiffReader(path) as reader:
        validate_mass_correction.verify(reader, outputs['full-mc'], outputs['sparse-mc'], {'offset_ppm': mc.offset_ppm})
    for kind in ('full', 'sparse'):
        verify_output(path, outputs[kind], kind, include_signature=False)


def test_validators_accept_native_output_without_using_the_extension(acquisition, tmp_path, monkeypatch):
    mc, outputs = production(acquisition, tmp_path, 'native')
    monkeypatch.setattr(native, '_native', Unusable())
    validators(acquisition, tmp_path, mc, outputs, convert=False)


@pytest.mark.parametrize('validator', ['convert', 'sparse', 'mass-correction', 'parity-full', 'parity-sparse'])
def test_validators_reject_a_wrong_native_kernel(acquisition, tmp_path, monkeypatch, validator):
    from scripts import validate_mass_correction, validate_sparse_profile
    monkeypatch.setattr(native, '_native', Wrong(native._native))
    mc, outputs = production(acquisition, tmp_path, 'wrong')
    verify_output = runpy.run_path(str(ROOT / 'scripts/integration-parity.py'))['verify_output']
    with pytest.raises(AssertionError):
        if validator == 'convert':
            validate_sparse_profile.convert(acquisition, tmp_path, 'converted')
        elif validator == 'sparse':
            with WiffReader(acquisition) as reader:
                reference = writer.write_mzml(reader, tmp_path / 'reference.mzML')
            validate_sparse_profile.compare(reference, outputs['sparse'])
        elif validator == 'mass-correction':
            with WiffReader(acquisition) as reader:
                validate_mass_correction.verify(reader, outputs['full-mc'], outputs['sparse-mc'], {'offset_ppm': mc.offset_ppm})
        else:
            kind = validator.removeprefix('parity-')
            verify_output(acquisition, outputs[kind], kind, include_signature=False)
    with pytest.raises(AssertionError, match='interpreted reference'):
        with native.NativeWiffReader(acquisition) as reader:
            validate_mass_correction.verify(reader, outputs['full-mc'], outputs['sparse-mc'], {'offset_ppm': mc.offset_ppm})


def test_the_wrong_kernel_really_changes_production_output(acquisition, tmp_path, monkeypatch):
    with native.NativeWiffReader(acquisition) as reader:
        right = writer.write_mzml(reader, tmp_path / 'right.mzML')
    monkeypatch.setattr(native, '_native', Wrong(native._native))
    with native.NativeWiffReader(acquisition) as reader:
        wrong = writer.write_mzml(reader, tmp_path / 'wrong.mzML')
    with WiffReader(acquisition) as reader:
        reference = writer.write_mzml(reader, tmp_path / 'reference.mzML')  # Never the (wrong) extension.
    assert right.read_bytes() == reference.read_bytes() != wrong.read_bytes()
