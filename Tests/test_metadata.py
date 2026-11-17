"""Public metadata validation must catch independently stale packaged components."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import plistlib

import pytest
from kmzcore.version import LICENSE_NOTICE, MINIMUM_MACOS, __version__

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('check_versions', ROOT / 'scripts/check_versions.py')
versions = module_from_spec(spec)
spec.loader.exec_module(versions)


def test_public_source_versions_agree():
    versions.check_source()


@pytest.fixture
def distribution(tmp_path):
    (tmp_path / 'build.json').write_text(json.dumps({
        'version': __version__, 'python': (ROOT / '.python-version').read_text().strip()}))
    (tmp_path / 'LICENSE.txt').write_bytes((ROOT / 'LICENSE').read_bytes())
    for name in versions.APPS:
        contents = tmp_path / (name + '.app') / 'Contents'
        (contents / 'Resources/docs').mkdir(parents=True)
        (contents / 'Resources/docs' / (name + '.html')).write_text('version ' + __version__)
        (contents / 'Resources/icon.icns').write_bytes(b'icns')
        (contents / 'Info.plist').write_bytes(plistlib.dumps({
            'CFBundleShortVersionString': __version__, 'CFBundleVersion': __version__,
            'CFBundleIconFile': 'icon.icns', 'LSMinimumSystemVersion': MINIMUM_MACOS,
            'NSHumanReadableCopyright': LICENSE_NOTICE}))
    return tmp_path


def test_stale_bundle_is_rejected(distribution):
    path = distribution / 'KmzViewer.app/Contents/Info.plist'
    info = plistlib.loads(path.read_bytes())
    info['CFBundleShortVersionString'] = '0.0.1'
    path.write_bytes(plistlib.dumps(info))
    with pytest.raises(ValueError, match='KmzViewer CFBundleShortVersionString'):
        versions.check_distribution(distribution)


@pytest.mark.parametrize('key, value', [('LSMinimumSystemVersion', '14.0'),
                                        ('NSHumanReadableCopyright', 'KmzSuite')])
def test_release_metadata_mismatch_is_rejected(distribution, key, value):
    path = distribution / 'KmzCoMet.app/Contents/Info.plist'
    path.write_bytes(plistlib.dumps(plistlib.loads(path.read_bytes()) | {key: value}))
    with pytest.raises(ValueError, match='KmzCoMet ' + key):
        versions.check_distribution(distribution)


def test_distribution_without_license_is_rejected(distribution, monkeypatch):
    monkeypatch.setattr(versions.subprocess, 'check_output', lambda command, **kwargs: __version__)
    (distribution / 'LICENSE.txt').write_text('other')
    with pytest.raises(ValueError, match='LICENSE.txt'):
        versions.check_distribution(distribution)


def test_stale_embedded_cli_is_rejected(distribution, monkeypatch):
    def output(command, **kwargs):
        return '0.0.1' if 'KmzConvert.app' in command[0] else __version__
    monkeypatch.setattr(versions.subprocess, 'check_output', output)
    with pytest.raises(ValueError, match='KmzConvert.app/Contents/Resources/Converter/kmzconvert'):
        versions.check_distribution(distribution)


@pytest.mark.parametrize('runtime', ['3.15.0', '3.1.4', '2.14.7'])
def test_distribution_python_outside_series_is_rejected(distribution, runtime):
    path = distribution / 'build.json'
    path.write_text(json.dumps(json.loads(path.read_text()) | {'python': runtime}))
    with pytest.raises(ValueError, match='distribution CPython runtime'):
        versions.check_distribution(distribution)


def test_any_patch_release_is_in_series():
    series = versions.python_series()
    assert versions.in_series(series + '.0', series) and versions.in_series(series + '.99', series)
