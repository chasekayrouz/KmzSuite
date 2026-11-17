"""Explicit scan sources; native WIFF follows the Full Profile writer contract."""
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Scan


@dataclass(frozen=True)
class InputSource:
    path: Path
    sample_id: int | None = None

    def resolved(self):
        path = Path(self.path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in {'.mzml', '.wiff', '.wiff2'}:
            raise ValueError('Choose local mzML or qualified WIFF acquisitions')
        if path.suffix.lower() == '.mzml':
            if self.sample_id is not None:
                raise ValueError('A native sample ID applies only to WIFF inputs')
            return InputSource(path)
        from kmzconvert_core.samples import list_samples
        ids = [sample.id for sample in list_samples(path)]
        selected = self.sample_id
        if selected is None:
            if len(ids) != 1:
                raise ValueError(f'{path.name}: select the stored sample IDs before analysis')
            selected = ids[0]
        if type(selected) is not int or selected not in ids:
            raise ValueError(f'{path.name}: unknown stored sample ID')
        return InputSource(path, selected)

    def files(self):
        if self.path.suffix.lower() == '.mzml':
            return [self.path]
        paths = [self.path if self.path.suffix.lower() == '.wiff' else self.path.with_suffix('.wiff'),
                 self.path.with_suffix('.wiff.scan')]
        if self.path.suffix.lower() == '.wiff2':
            paths.append(self.path)
        # WiffReader also inspects optional optical sidecars. They are not MS1
        # measurements, but consumed file identities still belong in provenance.
        paths.extend(p for p in (self.path.with_suffix('.dad.scan'), self.path.with_suffix('.dad.sidx')) if p.is_file())
        return paths

    def identity(self):
        paths = self.files()[:2]
        return tuple((p.stat().st_dev, p.stat().st_ino) for p in paths), self.sample_id


def validate_scan(scan, previous, arrays_valid=None):
    """``arrays_valid`` may carry ``ordered_finite`` of the scan's arrays, computed earlier."""
    if not np.isfinite(scan.rt) or scan.rt < 0 or scan.rt <= previous.get(scan.polarity, -1):
        raise ValueError('MS1 times must be finite and strictly increasing within each polarity')
    previous[scan.polarity] = scan.rt
    if not (ordered_finite(scan.mz, scan.intensity) if arrays_valid is None else arrays_valid):
        raise ValueError('Require ordered finite masses and nonnegative intensities')


def ordered_finite(m, y):
    """1-D arrays of one shape: finite masses that are positive and strictly increasing,
    and finite nonnegative intensities. In fewer passes: increasing masses compare
    every element (NaN fails), so the first above 0 and the last below infinity
    bound the rest; NaN propagates through min/max."""
    if m.ndim != 1 or m.shape != y.shape:
        return False
    return not len(m) or bool(m[0] > 0 and m[-1] < np.inf and np.all(m[1:] > m[:-1])
                              and y.min() >= 0 and y.max() < np.inf)


# Helper threads decoding the next spectra while one is consumed (the native
# kernel releases the GIL); more mostly contend for the GIL with the consumer.
DECODERS = 2


def native_scans(source, checkpoint=lambda: None, *, framing=True):
    """Validated MS1 scans of one stored WIFF sample through the converter's exact
    shared codec. Framed spectra are decoded, measured, guarded and framed in one
    native call (unframed spectra decode natively), with the interpreted reference
    wherever the extension declines or is absent. With the extension, ``DECODERS``
    threads decode ahead in order; closing joins them before the reader closes."""
    from kmzconvert_core import native
    previous = {}
    with native.NativeWiffReader(source.path, sample_id=source.sample_id) as reader:
        def records():
            for record in reader.scans:
                checkpoint()
                if reader.experiments[record.experiment-1].ms_level == 1:
                    yield record

        def decode(record):
            # Integer counts below 2**53 sum exactly even when zeros are omitted.
            # For unusually large totals retain the reference reduction layout.
            if framing or record.tic >= 2**53:
                mz, intensity = native.reader_arrays(reader, record)
            else:
                mz, intensity = reader.arrays(record, framing=False, decoder=native.decode_tokens)
            return record, mz, intensity, ordered_finite(mz, intensity)

        workers = DECODERS if native.ENGAGED else 0
        with closing(native.ordered(decode, records(), workers)) as decoded:
            for record, mz, intensity, arrays_valid in decoded:
                experiment = reader.experiments[record.experiment-1]
                polarity = {'positive': 1, 'negative': -1}.get(experiment.polarity, 0)
                scan = Scan(record.rt_seconds/60, polarity, 'profile', mz, intensity,
                            ((experiment.lower_mz, experiment.upper_mz),), 'number of detector counts')
                validate_scan(scan, previous, arrays_valid)
                yield scan


def eic_scans(sample, checkpoint=lambda: None):
    """All measured points, without the zeros added only to draw spectra."""
    if sample.path.suffix.lower() in {'.wiff', '.wiff2'}:
        return native_scans(InputSource(sample.path, sample.native_sample_id), checkpoint, framing=False)
    from .reader import iter_scans
    return iter_scans(sample.path)
