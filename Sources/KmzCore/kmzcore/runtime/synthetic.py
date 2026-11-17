"""Small public synthetic fixture for frozen-app validation; no acquired data."""
from pathlib import Path
import base64
import struct
import zlib


def write_fixture(path, *, include_dad=False):
    water = 18.01056468403
    scans = [(1, 1.0, [100., 200.], [5., 2.], 40.),
             (1, 2.0, [100., 200.], [8., 3.], 50.),
             (2, 2.5, [300. - water, 180., 300.], [80., 40., 10.], 130.)]
    blocks = []
    def cv(accession, name, value='', extra=''):
        return f'<cvParam cvRef="MS" accession="{accession}" name="{name}" value="{value}" {extra}/>'
    for index, (level, rt, mz, intensity, tic) in enumerate(scans):
        arrays = []
        for values, accession, name in [(mz, 'MS:1000514', 'm/z array'), (intensity, 'MS:1000515', 'intensity array')]:
            binary = base64.b64encode(zlib.compress(struct.pack('<' + 'd' * len(values), *values))).decode()
            arrays.append('<binaryDataArray>' + cv('MS:1000523', '64-bit float') + cv('MS:1000574', 'zlib compression') + cv(accession, name) + f'<binary>{binary}</binary></binaryDataArray>')
        precursor = '' if level == 1 else '<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>' + cv('MS:1000744', 'selected ion m/z', 300) + cv('MS:1000041', 'charge state', 1) + '</selectedIon></selectedIonList></precursor></precursorList>'
        blocks.append(f'<spectrum index="{index}" id="synthetic-{index}" defaultArrayLength="{len(mz)}">' + cv('MS:1000511', 'ms level', level) + cv('MS:1000127', 'centroid spectrum') + cv('MS:1000130', 'positive scan') + cv('MS:1000285', 'total ion current', tic) + '<scanList count="1"><scan>' + cv('MS:1000016', 'scan start time', rt, 'unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"') + '</scan></scanList>' + precursor + '<binaryDataArrayList count="2">' + ''.join(arrays) + '</binaryDataArrayList></spectrum>')
    if include_dad:
        for row, (rt, values) in enumerate(((1.1, [-.002, .01, .025, .008, -.001]), (1.9, [-.003, .015, .04, .01, -.002]))):
            arrays = []
            for data, accession, name, unit, unit_name in (([250., 252., 254., 256., 258.], 'MS:1000617', 'wavelength array', 'UO:0000018', 'nanometer'),
                                                         (values, 'MS:1000515', 'intensity array', 'UO:0000269', 'absorbance unit')):
                binary = base64.b64encode(zlib.compress(struct.pack('<5d', *data))).decode()
                arrays.append('<binaryDataArray>' + cv('MS:1000523', '64-bit float') + cv('MS:1000574', 'zlib compression') +
                              cv(accession, name, extra=f'unitCvRef="UO" unitAccession="{unit}" unitName="{unit_name}"') + f'<binary>{binary}</binary></binaryDataArray>')
            blocks.append(f'<spectrum index="{3+row}" id="optical-{row}" defaultArrayLength="5">' + cv('MS:1000806', 'absorption spectrum') +
                          cv('MS:1000128', 'profile spectrum') + '<scanList count="1"><scan>' +
                          cv('MS:1000016', 'scan start time', rt, 'unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"') +
                          '</scan></scanList><binaryDataArrayList count="2">' + ''.join(arrays) + '</binaryDataArrayList></spectrum>')
    Path(path).write_text(f'<?xml version="1.0"?><mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0"><run id="synthetic"><spectrumList count="{len(blocks)}">' + ''.join(blocks) + '</spectrumList></run></mzML>')
