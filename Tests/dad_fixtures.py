"""Generic synthetic optical acquisitions; no instrument-derived measurements."""
from pathlib import Path
from types import SimpleNamespace
import struct

import numpy as np

from kmzconvert_core.dad import DadData
from kmzconvert_core.reader import Experiment, Scan


def sidecars(count=5, width=11):
    times = .05 + np.arange(count) * .25
    wavelengths = 250. + np.arange(width)
    values = np.arange(count * width, dtype=float).reshape(count, width) * .125 - 3.
    total = values.sum(axis=1)
    prefix = bytes.fromhex('0000000004000000') + bytes(24)
    header = prefix + bytes(2) + struct.pack('<IIIffff', count, width, 1, times[-1], wavelengths[0], wavelengths[-1], 1.) + bytes(20) + struct.pack('<f', times[0])
    scan_size = 44 + values.nbytes
    scan = struct.pack('<IQIIIIIQI', 2200, scan_size, 0, 0, 0x11111111, 2200, 1, scan_size-44, 0) + values.astype('<f8').tobytes()
    index_size = 52 + 42 * count
    index = struct.pack('<8sIQIIIIQII', b'@dadsidx', 1002, index_size, 1, 0, 1002, 1, index_size, 0, 0)
    index += b''.join(struct.pack('<IQIHddd', i, i*width*8, width*8, 7, rt, twc, 0.) for i, (rt, twc) in enumerate(zip(times, total)))
    key = 'SampleSubtree/Sample1/Devices/Device_1/DAD/'
    streams = {key+'DADSampleHeader': header, key+'DADRealTimeData': prefix + values.astype('<f4').tobytes(),
               key+'DADIdx': prefix + np.column_stack((times, total)).astype('<f4').tobytes()}
    return scan, index, streams, DadData(wavelengths, times, values, total, [])


def acquisition(folder, name='sample', *, dad=True, offset=0.):
    folder = Path(folder)
    paths = [folder / (name + suffix) for suffix in ('.wiff', '.wiff.scan')]
    for path in paths:
        path.write_bytes(b'Synthetic source placeholder; no private acquisition')
    scans = [Scan(i, i+1, 1, rt*60, (i+1)*1000., (i+1)*1000., 0, 0, .1, .2, None, None)
             for i, rt in enumerate(np.arange(0., 1.51, .1) + offset)]
    data = sidecars()[3] if dad else None
    if data is not None:
        data.rt_min += offset
    return SimpleNamespace(entry=paths[0], wiff_path=paths[0], scan_path=paths[1], dad=data,
        acquisition_software='SCIEX OS 2.2.0.0', instrument={'Mass Spectrometer': 'X500 QTOF'},
        scans=scans, experiments=[Experiment(1, 'positive', 100., 500., 200., 5., {})],
        arrays=lambda s: (np.linspace(100, 500, 5), np.array([0., s.tic, s.tic*.25, s.tic*.5, 0.])),
        tic_arrays=lambda: (np.array([s.rt_seconds/60 for s in scans]), np.array([s.tic for s in scans]), np.ones(len(scans), dtype=int)))
