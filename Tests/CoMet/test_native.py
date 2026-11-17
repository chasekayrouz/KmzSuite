from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pytest

from kmzcore.cancel import Cancelled
from kmzcomet.engine import analyze
from kmzcomet.model import Settings
from kmzcomet.sources import InputSource,native_scans
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.writer import write_mzml
from Tests.Convert.test_samples import multi


def test_native_sample_selection_aliases_and_provenance(multi,tmp_path):
    path,_=multi
    with pytest.raises(ValueError,match='select the stored'):analyze([path])
    a=InputSource(path,2);b=InputSource(path,7)
    result=analyze([a,b])
    assert [s.native_sample_id for s in result.samples]==[2,7]
    assert all(len(s.source_files)==2 for s in result.samples)
    alias=tmp_path/'alias.wiff';os.link(path,alias);os.link(path.with_suffix('.wiff.scan'),alias.with_suffix('.wiff.scan'))
    with pytest.raises(ValueError,match='Duplicate'):analyze([a,InputSource(alias,2)])
    wiff2=path.with_suffix('.wiff2');wiff2.write_bytes(bytes(16)+bytes.fromhex('100001010c402020'))
    with pytest.raises(ValueError,match='Duplicate'):analyze([a,InputSource(wiff2,2)])


def test_native_cancellation_closes_the_reader(multi,monkeypatch):
    path,_=multi;readers=[]
    import kmzconvert_core.native as module
    original=module.NativeWiffReader
    class Tracked(original):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);readers.append(self)
    monkeypatch.setattr(module,'NativeWiffReader',Tracked)
    calls=0
    def checkpoint():
        nonlocal calls
        calls+=1
        if calls==1:raise Cancelled()
    with pytest.raises(Cancelled):list(native_scans(InputSource(path,2),checkpoint))
    assert readers and all(r._mapping.closed for r in readers)


def test_eic_extraction_preserves_nonconsecutive_stored_sample_ids(multi):
    from kmzcomet.chromatograms import ChromatogramCache
    from kmzcomet.model import Feature
    from kmzviewer.extract import ppm_window, sum_interval
    path,_=multi
    result=analyze([InputSource(path,2),InputSource(path,7)])
    scans=[list(native_scans(InputSource(path,ident))) for ident in (2,7)]
    mass=float(scans[0][0].mz[np.argmax(scans[0][0].intensity)])
    result.features=[Feature('F1',mass,.1,1,[],0.,.2)]
    batch=ChromatogramCache(result).extract([(0,0),(0,1)])
    assert [entry.native_sample_id for entry in batch.entries]==[2,7]
    low,high=ppm_window(mass,result.settings.mz_ppm)
    for entry,source in zip(batch.entries,scans):
        assert entry.data is not None
        np.testing.assert_array_equal(entry.data.rt,[s.rt for s in source])
        np.testing.assert_array_equal(entry.data.y,[sum_interval(s.mz,s.intensity,low,high) for s in source])
    assert batch.entries[0].data.y[0]!=batch.entries[1].data.y[0]


def test_native_metadata_change_discards_result(multi):
    path,_=multi
    def progress(message):
        if message=='Align candidates':path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='input changed'):analyze([InputSource(path,2)],progress=progress)


@pytest.mark.integration
def test_native_full_mzml_analysis_parity(tmp_path):
    root=Path(__file__).resolve().parents[2]/'TestData/Local'
    entries=[root/f'acquisition-{i:02}.wiff' for i in (1,2)]
    if not all(p.exists() for p in entries):pytest.skip('Optional qualified local acquisitions unavailable')
    converted=[]
    for i,entry in enumerate(entries):
        path=tmp_path/f'S{i}.mzML'
        with WiffReader(entry) as reader:write_mzml(reader,path,profile='full')
        converted.append(path)
    settings=Settings(mz_ppm=30,discovery_window=.35,min_contrast=2,min_height=100)
    native=analyze(entries,settings);mzml=analyze(converted,settings)
    assert [asdict(f) for f in native.features]==[asdict(f) for f in mzml.features]
    assert [[asdict(m) for m in col] for col in native.measurements]==[[asdict(m) for m in col] for col in mzml.measurements]
    for a,b in zip(native.samples,mzml.samples):
        assert a.scans==b.scans and a.shifts==b.shifts and a.anchors==b.anchors
        assert a.candidates==b.candidates and a.units==b.units and a.modes==b.modes
