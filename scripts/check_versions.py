#!/usr/bin/env python3
"""Fail on divergent public component, manual, runtime or packaged versions."""
import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import tomllib

from kmzcore.version import COPYRIGHT, LICENSE_NOTICE, MINIMUM_MACOS, __version__

ROOT = Path(__file__).resolve().parents[1]
APPS = ('KmzSuite', 'KmzViewer', 'Kmz2Viewer', 'KmzConvert', 'KmzCoMet')
MANUAL_VERSION = re.compile(r'\bversion (\d+\.\d+\.\d+)\b')


def equal(label, actual, expected=__version__):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected}, found {actual!r}')


def python_series():
    return (ROOT / '.python-version').read_text().strip()


def in_series(version, series):
    return version.split('.')[:len(series.split('.'))] == series.split('.')


def check_manual(path):
    versions = MANUAL_VERSION.findall(path.read_text(encoding='utf-8'))
    if not versions:
        raise ValueError(f'{path.name}: missing public version')
    for version in versions:
        equal(path.name, version)


def check_source():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    equal('Python package metadata', metadata.version('kmzsuite'))
    major, minor = map(int, python_series().split('.')[:2])
    equal('Python runtime requirement', project['project']['requires-python'],
          f'>={major}.{minor},<{major}.{minor + 1}')
    for name in ('kmzcore', 'kmzviewer', 'kmz2viewer', 'kmzconvert_core', 'kmzconvert', 'kmzcomet', 'kmzcomet_app'):
        equal(name, import_module(name).__version__)
    cli = subprocess.check_output([sys.executable, '-m', 'kmzconvert', '--version'], text=True).strip()
    equal('development kmzconvert --version', cli)
    cli = subprocess.check_output([sys.executable, '-m', 'kmzcomet', '--version'], text=True).strip()
    equal('development kmzcomet --version', cli)
    for name in APPS:
        check_manual(ROOT / 'docs' / (name + '.html'))
    if COPYRIGHT.replace('©', '(c)') not in (ROOT / 'LICENSE').read_text():
        raise ValueError(f'LICENSE: expected the notice {COPYRIGHT!r}')


def check_distribution(directory):
    manifest = json.loads((directory / 'build.json').read_text())
    equal('distribution manifest', manifest['version'])
    if not in_series(manifest['python'], python_series()):
        raise ValueError(f"distribution CPython runtime: expected {python_series()}.x, found {manifest['python']!r}")
    for name in APPS:
        contents = directory / (name + '.app') / 'Contents'
        info = plistlib.loads((contents / 'Info.plist').read_bytes())
        for key in ('CFBundleShortVersionString', 'CFBundleVersion'):
            equal(name + ' ' + key, info[key])
        equal(name + ' LSMinimumSystemVersion', info.get('LSMinimumSystemVersion'), MINIMUM_MACOS)
        equal(name + ' NSHumanReadableCopyright', info.get('NSHumanReadableCopyright'), LICENSE_NOTICE)
        check_manual(contents / 'Resources/docs' / (name + '.html'))
        icon = info.get('CFBundleIconFile', '')
        icon_path = contents / 'Resources' / icon
        if not icon_path.is_file() or icon_path.read_bytes()[:4] != b'icns':
            raise ValueError(f'{name}: missing or invalid bundle icon')
    if (directory / 'LICENSE.txt').read_bytes() != (ROOT / 'LICENSE').read_bytes():
        raise ValueError('distribution LICENSE.txt differs from LICENSE')
    for cli in (directory / 'kmzconvert', directory / 'KmzConvert.app/Contents/Resources/Converter/kmzconvert'):
        equal(str(cli.relative_to(directory)), subprocess.check_output([str(cli.resolve()), '--version'], text=True).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path)
    parser.add_argument('--update-docs', action='store_true', help='Synchronize manual version labels from the canonical version')
    args = parser.parse_args()
    if args.update_docs:
        for name in APPS:
            path = ROOT / 'docs' / (name + '.html')
            path.write_text(MANUAL_VERSION.sub('version ' + __version__, path.read_text(encoding='utf-8')), encoding='utf-8')
    try:
        check_source()
        if args.distribution:
            check_distribution(args.distribution.resolve())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Version/metadata validation failed: {error}') from error
    print(f'Public versions agree: {__version__}' + (' (including distribution)' if args.distribution else ''))


if __name__ == '__main__':
    main()
