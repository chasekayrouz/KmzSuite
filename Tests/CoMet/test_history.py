import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest

from kmzcore.cancel import Cancelled
from kmzcomet.engine import analyze
from kmzcomet.csv_export import write_csv
from kmzcomet.excel import write_workbook
from kmzcomet.history import load_analysis, payload, restore, save_analysis
from kmzcomet.model import Settings
from scripts.benchmarks.comet_fixtures import Truth, write_run


@pytest.fixture
def result(tmp_path):
    truth=[Truth('shared',200.,1.,(9000.,600.)),Truth('only-first',300.,1.5,(8000.,0.))]
    paths=[write_run(tmp_path/f'sample-{i}.mzML',truth,sample=i,noise=0.,rt_stop=2.) for i in range(2)]
    return analyze(paths,Settings(alignment='none'))


def test_snapshot_reopens_without_sources_and_retains_original_metadata(result,tmp_path):
    original=payload(result)
    path=save_analysis(result,tmp_path/'history')
    for sample in result.samples:sample.path.unlink()
    reopened=load_analysis(path)
    assert payload(reopened)==original
    assert reopened.samples[0].shifts=={1:0.}
    assert save_analysis(reopened,path.parent)==path
    xlsx=write_workbook(reopened,tmp_path/'reopened.xlsx')
    book=load_workbook(xlsx,read_only=True)
    run=list(book['Run'].iter_rows(values_only=True))
    assert ('Analysis','Created UTC',result.created_utc) in run
    assert ('Analysis','Analysis ID',result.analysis_id) in run
    assert ('Analysis','Method',result.method) in run
    assert ('Analysis','Suite version',result.suite_version) in run
    book.close()


def test_csv_preserves_full_matrix_and_zero_missing_and_formula_safe_headers(result,tmp_path):
    result.samples[0].path=Path('=formula.mzML')
    # An unacquired pair must stay blank, unlike a measured zero.
    result.measurements[1][0]=replace(result.measurements[1][0],status='not_acquired',height=None)
    path=write_csv(result,tmp_path/'matrix.csv')
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.reader(stream))
    assert len(rows)==len(result.features)+1
    assert rows[0][4].startswith('S001: =formula')
    assert rows[1][5]=='' and float(rows[2][5])==0.
    with pytest.raises(FileExistsError):write_csv(result,path)


@pytest.mark.parametrize('change',["schema","nan","dimensions","references","duplicate_id","candidate","polarity_keys","bad_type"])
def test_reject_malformed_snapshot(result,tmp_path,change):
    data=payload(result);a=data['analysis']
    if change=='schema':data['schema']=99
    elif change=='nan':a['features'][0]['mz']=float('nan')
    elif change=='dimensions':a['measurements'][0].pop()
    elif change=='references':a['measurements'][0][0]['feature_id']='other'
    elif change=='duplicate_id':a['features'][1]['id']=a['features'][0]['id']
    elif change=='candidate':a['features'][0]['members'][0]['sample']=99
    elif change=='polarity_keys':a['samples'][0]['shifts']={'positive':0.}
    elif change=='bad_type':a['measurements'][0][0]['points']=True
    path=tmp_path/'invalid.kmzcomet'
    with ZipFile(path,'w') as z:z.writestr('analysis.json',json.dumps(data))
    with pytest.raises(ValueError,match='Could not open'):load_analysis(path)


def test_truncated_archive_duplicate_json_and_decompression_limit(result,tmp_path,monkeypatch):
    path=tmp_path/'bad.kmzcomet';path.write_bytes(b'PK\x03')
    with pytest.raises(ValueError,match='Could not open'):load_analysis(path)
    with ZipFile(path,'w') as z:z.writestr('analysis.json','{"schema":1,"schema":1,"analysis":{}}')
    with pytest.raises(ValueError,match='duplicate JSON key'):load_analysis(path)
    good=save_analysis(result,tmp_path/'history')
    monkeypatch.setattr('kmzcomet.history.MAX_JSON',10)
    with pytest.raises(ValueError,match='size limit'):load_analysis(good)


def test_cancelled_save_and_failure_never_publish_or_leave_staging(result,tmp_path,monkeypatch):
    def stop():raise Cancelled()
    with pytest.raises(Cancelled):save_analysis(result,tmp_path/'history',checkpoint=stop)
    assert not list((tmp_path/'history').glob('*'))
    def failure(*args):raise OSError('disk full')
    monkeypatch.setattr('kmzcomet.output.publish',failure)
    with pytest.raises(OSError,match='disk full'):save_analysis(result,tmp_path/'history')
    assert list((tmp_path/'history').iterdir())==[]


def test_late_cancel_does_not_hide_committed_snapshot(result,tmp_path,monkeypatch):
    import kmzcomet.output as output
    original=output.publish;cancelled=False
    def publish(*args):
        nonlocal cancelled
        original(*args);cancelled=True
    def checkpoint():
        if cancelled:raise Cancelled()
    monkeypatch.setattr(output,'publish',publish)
    saved=save_analysis(result,tmp_path/'history',checkpoint=checkpoint)
    assert cancelled and payload(load_analysis(saved))==payload(result)


def test_publication_fallback_is_no_clobber(result,tmp_path,monkeypatch):
    import errno
    import kmzcomet.output as output
    def no_links(*args):raise OSError(errno.EOPNOTSUPP,'no hard links')
    monkeypatch.setattr(output.os,'link',no_links)
    path=write_csv(result,tmp_path/'fallback.csv')
    assert path.read_text(encoding='utf-8-sig').startswith('Feature ID')
    with pytest.raises(FileExistsError):write_csv(result,path)
    assert not list(tmp_path.glob('.kmzcomet-*'))
