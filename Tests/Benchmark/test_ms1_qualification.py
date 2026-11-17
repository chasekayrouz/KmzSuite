"""Qualification mechanics; synthetic values are never held-out evidence."""
from pathlib import Path

import numpy as np
import pytest

from kmzconvert_core import writer
from kmzconvert_core.reader import WiffReader
from kmzconvert_core.reference import iter_records
from scripts.benchmarks.ms1_algorithms import guard_indices
from scripts.benchmarks.ms1_qualification import evaluate, exact_subset, gate_file, inventory, summarize, write_d


def test_inventory_records_both_members_and_rejects_orphan(tmp_path):
    for i in range(5):
        (tmp_path/f'pair-{i}.wiff').write_bytes(b'entry'+bytes([i]))
        (tmp_path/f'pair-{i}.wiff.scan').write_bytes(b'scan'+bytes([i]))
    paths, rows=inventory(tmp_path)
    assert len(paths)==5 and len(rows)==10
    assert len({r['sha256'] for r in rows})==10
    assert all(r['pair']==paths[i//2].name for i,r in enumerate(rows))
    (tmp_path/'pair-4.wiff.scan').rename(tmp_path/'orphan.wiff.scan')
    with pytest.raises(ValueError,match='Orphaned'):
        inventory(tmp_path)


def test_exact_fidelity_measures_actual_decoded_values_and_rejects_duplicates():
    m=np.array([100.,101.,102.]);y=np.array([0.,101.,0.]);idx=np.arange(3)
    assert set(exact_subset(m,y,idx,m,y).values())=={0.}
    with pytest.raises(AssertionError,match='preservation'):
        exact_subset(m,y,idx,m,y+1)
    with pytest.raises(AssertionError,match='Duplicate'):
        exact_subset(m,y,np.array([1,1]),m[[1,1]],y[[1,1]])


@pytest.mark.parametrize('values', [[], [100], [101], [0,101,0], [101,0,0,101], [0,101,0,101,0]])
def test_frozen_qualification_edge_cases_and_guard_bound(values):
    y=np.asarray(values,dtype=float);m=(10+np.arange(len(y))*.0001)**2
    idx=guard_indices(y)
    row,_=evaluate(m,y,m[idx],y[idx],1)
    assert row['above_threshold_retained']==np.count_nonzero(y>100)
    assert row['added_guard_points']<=2*row['retained_regions']
    assert row['mz_max_absolute_deviation']==row['intensity_max_absolute_deviation']==0
    assert row['no_above_threshold']==int(not np.any(y>100))


def test_native_d_file_gate_and_production_patch_restoration(tmp_path):
    path=Path(__file__).resolve().parents[2]/'TestData/Local/acquisition-01.wiff'
    if not path.exists():
        pytest.skip('Optional real complete WIFF fixture absent')
    original_metadata,original_spectrum=writer.metadata,writer.spectrum
    with WiffReader(path) as reader:
        reader.scans=reader.scans[:3]
        ref,d=tmp_path/'reference.mzML',tmp_path/'D.mzML'
        writer.write_mzml(reader,ref)
        write_d(reader,d)
        import xml.etree.ElementTree as ET
        from kmzconvert_core.reference import NS
        spectrum=ET.parse(d).find('.//'+NS+'spectrum')
        names=[node.tag[len(NS):] for node in spectrum]
        assert names.index('userParam') < names.index('scanList')
        schema=Path(__file__).resolve().parents[2]/'.cache/standards/mzML1.1.0_idx.xsd'
        if schema.exists():
            from lxml import etree
            etree.XMLSchema(etree.parse(str(schema))).assertValid(etree.parse(str(d)))
        assert not any(r['accepted'] for r in gate_file(ref))
        gates=gate_file(d)
        assert all(r['accepted']==(r['ms_level']==1) for r in gates)
        for scan,record in zip(reader.scans,iter_records(d),strict=True):
            m,y=reader.arrays(scan)
            idx=guard_indices(y) if reader.experiments[scan.experiment-1].ms_level==1 else np.arange(len(m))
            exact_subset(m,y,idx,record['arrays']['mz'],record['arrays']['intensity'])
        with pytest.raises(FileExistsError):
            write_d(reader,d)
    assert writer.metadata is original_metadata and writer.spectrum is original_spectrum


def test_aggregate_of_numpy_peak_counts_is_json_serializable():
    import json
    m=np.array([100.,100.001,100.002]);y=np.array([0.,200.,0.]);idx=guard_indices(y)
    row,_=evaluate(m,y,m[idx],y[idx],1)
    row.update(acquisition='heldout-01',condition='D',source_signal_breaks=0,gate_accepted=1,gate_declined=0)
    result=summarize([row],'pooled','D')
    assert json.loads(json.dumps(result))['major_apices_retained']==1
