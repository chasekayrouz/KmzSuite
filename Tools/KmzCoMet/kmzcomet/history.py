"""Versioned, data-only result snapshots. Reopening never touches raw inputs."""
from dataclasses import fields, is_dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from uuid import UUID
from zipfile import ZipFile, ZIP_DEFLATED

from .model import Candidate, Feature, Measurement, Result, Sample, Settings
from .output import new_output

SCHEMA = 1
MAX_JSON = 512 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024
STATUSES = {'supported', 'weak_signal', 'no_signal', 'not_acquired', 'insufficient_coverage', 'ambiguous'}


def history_directory():
    override = os.environ.get('KMZ_LOCAL_STATE_ROOT')
    if override:
        return Path(override) / 'KmzCoMet' / 'Analyses'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'KmzSuite' / 'KmzCoMet' / 'Analyses'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local/share')) / 'KmzSuite' / 'KmzCoMet' / 'Analyses'


_ATOMS = {str, int, float, bool, type(None)}
_FIELDS, _NAMES = {}, {}


def _plain(value):
    """``asdict`` and JSON-ready conversion in one walk: dataclasses become dicts in
    field order, paths strings, sets sorted lists, dict keys strings, tuples lists."""
    kind = type(value)
    if kind in _ATOMS:
        return value
    names = _FIELDS.get(kind)
    if names is None:
        names = _FIELDS[kind] = tuple(f.name for f in fields(kind)) if is_dataclass(kind) else False
    if names is not False:
        return {name: _plain(getattr(value, name)) for name in names}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def payload(result):
    return {'schema': SCHEMA, 'analysis': _plain(result)}


def _require(condition, message):
    if not condition:
        raise ValueError('Invalid KmzCoMet analysis: ' + message)


def _record(value, cls):
    names = _NAMES.get(cls)
    if names is None:
        names = _NAMES[cls] = frozenset(f.name for f in fields(cls))
    _require(type(value) is dict and set(value) == names, f'{cls.__name__} fields')
    return value


def _number(value, *, minimum=None, optional=False):
    if optional and value is None:
        return
    _require(type(value) in (int, float) and math.isfinite(value), 'non-finite or non-numeric value')
    if minimum is not None:
        _require(value >= minimum, 'numeric value outside its range')


def _text(value, limit=4096):
    _require(type(value) is str and '\0' not in value and len(value) <= limit, 'invalid text')


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, 'invalid integer')


def _polarity(value):
    _require(type(value) is int and value in (-1, 0, 1), 'invalid polarity')


def _candidate(value, count):
    d = _record(value, Candidate)
    _integer(d['sample'])
    _require(d['sample'] < count, 'candidate sample reference')
    _number(d['mz'], minimum=1e-300)
    _number(d['rt'], minimum=0)
    _number(d['height'], minimum=0)
    _polarity(d['polarity'])
    return Candidate(**d)


def restore(envelope):
    _require(type(envelope) is dict and set(envelope) == {'schema', 'analysis'}, 'snapshot envelope')
    _require(type(envelope['schema']) is int and envelope['schema'] == SCHEMA, 'unsupported schema version')
    d = dict(_record(envelope['analysis'], Result))
    try:
        _require(type(d['analysis_id']) is str and str(UUID(d['analysis_id'])) == d['analysis_id'], 'analysis identifier')
        created = datetime.fromisoformat(d['created_utc'])
        _require(created.tzinfo is not None, 'analysis timestamp timezone')
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError('Invalid KmzCoMet analysis: identifier or timestamp') from error
    for key in ('method', 'suite_version'):
        _text(d[key], 100)
    _number(d['elapsed_seconds'], minimum=0)
    _record(d['settings'], Settings)
    settings = Settings(**d['settings'])
    _require(type(d['samples']) is list and 0 < len(d['samples']) <= 16380, 'sample count')
    samples = []
    for raw in d['samples']:
        value = dict(_record(raw, Sample))
        for key in ('path', 'id', 'sha256'):
            _text(value[key])
        _require(value['path'] and value['id'] and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'sample identity')
        _integer(value['size_bytes'])
        _integer(value['scans'], 1)
        _number(value['rt_min'], minimum=0)
        _number(value['rt_max'], minimum=value['rt_min'])
        _require(type(value['modes']) is list and len(value['modes']) == 1 and value['modes'][0] in {'profile', 'centroid'}, 'representation')
        _require(type(value['units']) is list and value['units'] and all(type(u) is str for u in value['units']), 'intensity units')
        _require(type(value['warnings']) is list, 'sample warnings')
        for warning in value['warnings']:
            _text(warning)
        for key in ('shifts', 'anchors', 'alignment_mad'):
            table = value[key]
            _require(type(table) is dict and set(table) <= {'-1', '0', '1'}, 'polarity diagnostic keys')
            value[key] = {int(k): v for k, v in table.items()}
        _require(set(value['shifts']) == set(value['anchors']) == set(value['alignment_mad']), 'alignment diagnostic dimensions')
        for shift in value['shifts'].values():
            _number(shift)
        for count in value['anchors'].values():
            _integer(count)
        for mad in value['alignment_mad'].values():
            _number(mad, minimum=0, optional=True)
        if value['native_sample_id'] is not None:
            _integer(value['native_sample_id'], 1)
        _require(type(value['source_files']) is list, 'input provenance')
        for source in value['source_files']:
            _require(type(source) is dict and set(source) == {'path', 'name', 'size_bytes', 'sha256'}, 'source file fields')
            _text(source['path']); _text(source['name'])
            _integer(source['size_bytes'])
            _require(type(source['sha256']) is str and re.fullmatch('[0-9a-f]{64}', source['sha256']), 'source digest')
        _require(type(value['candidates']) is list, 'candidate list')
        value['candidates'] = [_candidate(c, len(d['samples'])) for c in value['candidates']]
        _require(all(c.sample == len(samples) for c in value['candidates']), 'candidate ownership')
        value.update(path=Path(value['path']), modes=set(value['modes']), units=set(value['units']))
        samples.append(Sample(**value))
    _require(len({s.id for s in samples}) == len(samples), 'duplicate sample IDs')
    _require(type(d['features']) is list and len(d['features'])*len(samples)+1 <= 1048576, 'feature dimensions')
    features = []
    candidate_sets = [set(s.candidates) for s in samples]
    for raw in d['features']:
        value = dict(_record(raw, Feature))
        _text(value['id'])
        _require(bool(value['id']), 'empty feature ID')
        _number(value['mz'], minimum=1e-300)
        for key in ('rt', 'left', 'right'):
            _number(value[key])
        _require(value['left'] <= value['rt'] <= value['right'], 'feature bounds')
        _polarity(value['polarity'])
        _require(type(value['ambiguous']) is bool and type(value['members']) is list, 'feature metadata')
        value['members'] = [_candidate(c, len(samples)) for c in value['members']]
        _require(value['members'] and len({c.sample for c in value['members']}) == len(value['members']), 'feature discovery members')
        _require(all(c in candidate_sets[c.sample] and c.polarity == value['polarity'] for c in value['members']), 'feature candidate references')
        features.append(Feature(**value))
    _require(len({f.id for f in features}) == len(features), 'duplicate feature IDs')
    columns = d['measurements']
    _require(type(columns) is list and len(columns) == len(samples), 'measurement columns')
    measurements = []
    for sample, column in zip(samples, columns):
        _require(type(column) is list and len(column) == len(features), 'measurement rows')
        restored = []
        for feature, raw in zip(features, column):
            value = _record(raw, Measurement)
            _require(value['sample_id'] == sample.id and value['feature_id'] == feature.id, 'measurement references')
            _require(type(value['discovered']) is bool and value['status'] in STATUSES, 'measurement evidence')
            for key in ('height', 'height_above_background', 'window_area', 'apex_rt', 'apex_mz', 'background', 'noise', 'snr', 'contrast'):
                _number(value[key], minimum=0, optional=True)
            _integer(value['points']); _integer(value['support_points'])
            _require(value['support_points'] <= value['points'], 'scan support')
            _require((value['height'] is None) == (value['status'] == 'not_acquired'), 'missing versus measured height')
            _text(value['flags'], 16384)
            restored.append(Measurement(**value))
        measurements.append(restored)
    return Result(settings, samples, features, measurements, d['elapsed_seconds'],
                  d['analysis_id'], d['created_utc'], d['method'], d['suite_version'])


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def load_analysis(path):
    path = Path(path)
    try:
        _require(path.stat().st_size <= MAX_ARCHIVE, 'archive exceeds size limit')
        with ZipFile(path) as archive:
            _require(archive.namelist() == ['analysis.json'], 'archive members')
            _require(archive.getinfo('analysis.json').file_size <= MAX_JSON, 'analysis exceeds size limit')
            with archive.open('analysis.json') as handle:
                content = handle.read(MAX_JSON+1)
            _require(len(content) <= MAX_JSON, 'analysis exceeds size limit')
        envelope = json.loads(content, object_pairs_hook=_pairs,
                              parse_constant=lambda text: _require(False, 'non-finite JSON'))
        return restore(envelope)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
        raise ValueError(f'Could not open KmzCoMet analysis: {error}') from error
    except Exception as error:
        raise ValueError(f'Could not open KmzCoMet analysis: {error}') from error


def save_analysis(result, directory=None, *, checkpoint=lambda: None):
    directory = Path(directory) if directory is not None else history_directory()
    data = payload(result)
    restore(data)  # Never commit a snapshot that this schema cannot reopen.
    content = json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
    _require(len(content) <= MAX_JSON, 'analysis exceeds size limit')
    stamp = datetime.fromisoformat(result.created_utc).strftime('%Y-%m-%d_%H%M%S')
    path = directory / f'{stamp}_{len(result.samples)}-samples_{result.analysis_id}.kmzcomet'
    checkpoint()
    if path.exists():
        if payload(load_analysis(path)) == data:
            return path
        raise FileExistsError('An analysis with this identifier already exists')
    with new_output(path, checkpoint=checkpoint) as temporary:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as archive:
            archive.writestr('analysis.json', content)
        _require(temporary.stat().st_size <= MAX_ARCHIVE, 'archive exceeds size limit')
    # Publication is the commit point. A later cancel must not report failure.
    return path
