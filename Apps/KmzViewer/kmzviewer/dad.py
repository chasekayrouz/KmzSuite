"""Wavelength extraction from native optical scans, in nm and mAU."""
from dataclasses import dataclass
import numpy as np

from kmzcore.cancel import Cancelled
from kmzviewer.model import Trace
from kmzviewer.spectra import SpectrumReader, read_ahead, read_arrays


@dataclass(frozen=True)
class WavelengthBand:
    wavelength: float
    bandwidth: float

    def __post_init__(self):
        if not np.isfinite([self.wavelength, self.bandwidth]).all() or self.bandwidth < 0 or self.lower <= 0:
            raise ValueError("Wavelength must be positive; bandwidth must be finite and nonnegative.")

    @property
    def lower(self):
        return self.wavelength - self.bandwidth / 2

    @property
    def upper(self):
        return self.wavelength + self.bandwidth / 2

    def mean(self, wavelength, values):
        if not wavelength.size or self.lower < wavelength[0] or self.upper > wavelength[-1]:
            raise ValueError(f"Band {self.lower:g}–{self.upper:g} nm extends outside the acquired wavelength range.")
        selected = (wavelength >= self.lower) & (wavelength <= self.upper)
        if not selected.any():
            raise ValueError(f"No acquired wavelengths in {self.lower:g}–{self.upper:g} nm. Increase the bandwidth.")
        return float(np.mean(values[selected], dtype=np.float64))


@dataclass(frozen=True)
class WavelengthExtraction:
    signal: WavelengthBand | None = None  # None is TWC: every acquired wavelength.
    reference: WavelengthBand | None = None

    def __post_init__(self):
        if self.signal is None and self.reference is not None:
            raise ValueError("A TWC has no reference subtraction.")

    @property
    def label(self):
        if self.signal is None:
            return 'TWC'
        text = f'EWC {self.signal.wavelength:g} nm / {self.signal.bandwidth:g} nm'
        if self.reference is not None:
            text += f' · Ref {self.reference.wavelength:g} / {self.reference.bandwidth:g} nm'
        return text

    @property
    def detail(self):
        if self.signal is None:
            return 'Sum of absorbance at every acquired wavelength; mAU, signed native values.'
        text = f'Mean absorbance in {self.signal.lower:g}–{self.signal.upper:g} nm (inclusive)'
        if self.reference is not None:
            text += f' minus mean in {self.reference.lower:g}–{self.reference.upper:g} nm (inclusive)'
        return text + '; mAU. No interpolation, clipping, smoothing or baseline correction.'

    @property
    def extraction_key(self):
        if self.signal is None:
            return ('twc',)
        reference = None if self.reference is None else (float(self.reference.wavelength), float(self.reference.bandwidth))
        return ('ewc', float(self.signal.wavelength), float(self.signal.bandwidth), reference)


def extract_wavelengths(path, scans, settings, trace_ids, color, *, cache=None, on_progress=None, is_cancelled=None):
    if scans is None or not len(scans):
        raise ValueError('This file contains no DAD spectra. Reconvert with the DAD sidecars to include them.')
    if len(settings) != len(trace_ids):
        raise ValueError('Trace IDs do not match the wavelength extraction settings.')
    values = np.empty((len(settings), len(scans)))
    with SpectrumReader(path) as reader:
        read_ahead(reader, cache, path, scans.spectrum_index)
        for row, index in enumerate(scans.spectrum_index):
            if is_cancelled and is_cancelled():
                raise Cancelled()
            x, y = read_arrays(cache, path, int(index), lambda: reader.read(int(index)))
            if not x.size:
                raise ValueError(f'Empty DAD spectrum at {scans.rt_min[row]:g} min; no absorbance measurement is available.')
            for j, setting in enumerate(settings):
                result = float(np.sum(y, dtype=np.float64)) if setting.signal is None else setting.signal.mean(x, y)
                if setting.reference is not None:
                    result -= setting.reference.mean(x, y)
                values[j, row] = result
            if on_progress and (row % 10 == 0 or row + 1 == len(scans)):
                on_progress(row + 1, len(scans))
    if is_cancelled and is_cancelled():
        raise Cancelled()
    return [Trace(ident, 0, 'twc' if setting.signal is None else 'ewc', setting.label,
                  scans.rt_min.copy(), y, np.arange(len(scans)), setting.detail, color,
                  extraction_key=setting.extraction_key)
            for ident, setting, y in zip(trace_ids, settings, values)]
