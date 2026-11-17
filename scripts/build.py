#!/usr/bin/env python3
"""Freeze proven Python components and assemble the two native app bundles."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
from kmzcore.version import LICENSE_NOTICE, MINIMUM_MACOS, __version__

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    subprocess.run([str(p) for p in command], check=True, **kwargs)


def main():
    series = (ROOT / '.python-version').read_text().strip().split('.')
    if (sys.platform != 'darwin' or platform.machine() != 'arm64'
            or platform.python_implementation() != 'CPython'
            or platform.python_version().split('.')[:len(series)] != series):
        raise SystemExit(f'The packaging baseline is macOS arm64 / CPython {".".join(series)}.')
    from kmzconvert_core.native import ACCELERATED
    if not ACCELERATED:
        raise SystemExit('Shared native WIFF codec (KmzConvert/KmzCoMet) was not built; rerun the root build with the C toolchain available.')
    from kmzviewer.mzml_direct import ACCELERATED as VIEWER_ACCELERATED
    if not VIEWER_ACCELERATED:
        raise SystemExit('KmzViewer native mzML decoder was not built; rerun the root build with the C toolchain available.')
    from kmzcore.qualified import QUALIFIED, installed
    differing = [f'{name} {installed(name)} (qualified {pin})' for name, pin in QUALIFIED.items() if installed(name) != pin]
    if differing:
        raise SystemExit('Exact fast paths would be disabled in the apps; requalify them first (see kmzcore/qualified.py): ' + ', '.join(differing))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage = ROOT / 'build' / stamp
    products = stage / 'products'
    products.mkdir(parents=True)
    assets = stage / 'assets'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('QT_', 'DYLD_'))}
    env.update(QT_QPA_PLATFORM='offscreen', KMZ_BUILD_ASSETS=str(assets))
    run([sys.executable, '-c', 'from pathlib import Path; import sys; sys.path.insert(0, "packaging"); from assets import generate; generate(Path.cwd(), Path(sys.argv[1]))', assets], cwd=ROOT, env=env)
    for key, name in [('ms1', 'KmzViewer'), ('ms2', 'Kmz2Viewer'), ('comet', 'KmzCoMet'), ('convert', 'kmzconvert')]:
        env['KMZ_BUILD_APP'] = key
        spec = 'converter.spec' if key == 'convert' else 'comet.spec' if key == 'comet' else 'viewer.spec'
        log = stage / (name + '.log')
        with log.open('w') as handle:
            run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--distpath', stage / key,
                 '--workpath', stage / ('work-' + key), ROOT / 'packaging' / spec], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        source = stage / key / (name if key == 'convert' else name + '.app')
        shutil.move(source, products / source.name)
        print('Built', name, flush=True)
    runtime = products / 'kmzconvert'
    runtime.rename(products / 'Converter')
    (products / 'kmzconvert').write_text('#!/bin/sh\nKMZ_BIN_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\nexec "$KMZ_BIN_DIR/Converter/kmzconvert" "$@"\n')
    (products / 'kmzconvert').chmod(0o755)
    for name, identifier in [('KmzSuite', 'org.kmzsuite.launcher'), ('KmzConvert', 'org.kmzsuite.convert')]:
        app = products / (name + '.app')
        contents = app / 'Contents'
        for folder in ('MacOS', 'Resources/docs', 'Frameworks'):
            (contents / folder).mkdir(parents=True)
        shutil.copy2(ROOT / '.build/release' / name, contents / 'MacOS' / name)
        shutil.copy2(ROOT / 'docs' / (name + '.html'), contents / 'Resources/docs' / (name + '.html'))
        shutil.copy2(ROOT / 'LICENSE', contents / 'Resources/LICENSE')
        info = dict(CFBundleIdentifier=identifier, CFBundleName=name, CFBundleDisplayName=name,
                    CFBundleExecutable=name, CFBundlePackageType='APPL', CFBundleVersion=__version__,
                    CFBundleShortVersionString=__version__, LSMinimumSystemVersion=MINIMUM_MACOS, NSHighResolutionCapable=True,
                    NSHumanReadableCopyright=LICENSE_NOTICE, NSPrincipalClass='NSApplication')
        icon = 'suite' if name == 'KmzSuite' else 'convert'
        shutil.copy2(assets / (icon + '.icns'), contents / 'Resources' / (name + '.icns'))
        info['CFBundleIconFile'] = name + '.icns'
        if name == 'KmzConvert':
            shutil.copytree(products / 'Converter', contents / 'Resources/Converter', symlinks=True)
            shutil.copytree(assets / 'THIRD_PARTY_LICENSES', contents / 'Resources/THIRD_PARTY_LICENSES')
            shutil.copy2(ROOT / 'resources/THIRD_PARTY_NOTICES.txt', contents / 'Resources/THIRD_PARTY_NOTICES.txt')
            info['CFBundleDocumentTypes'] = [dict(CFBundleTypeName='SCIEX acquisition entry point',
                CFBundleTypeExtensions=['wiff', 'wiff2'], CFBundleTypeRole='Editor', LSHandlerRank='Alternate')]
        with (contents / 'Info.plist').open('wb') as handle:
            plistlib.dump(info, handle)
        run(['/usr/bin/codesign', '--force', '--deep', '--sign', '-', '--identifier', identifier, app])
    for name in ('KmzSuite', 'KmzViewer', 'Kmz2Viewer', 'KmzConvert', 'KmzCoMet'):
        run(['/usr/bin/codesign', '--verify', '--deep', '--strict', products / (name + '.app')])
    shutil.copytree(assets / 'THIRD_PARTY_LICENSES', products / 'licenses')
    shutil.copy2(ROOT / 'resources/THIRD_PARTY_NOTICES.txt', products / 'THIRD_PARTY_NOTICES.txt')
    shutil.copy2(ROOT / 'LICENSE', products / 'LICENSE.txt')
    (products / 'build.json').write_text(json.dumps({'version': __version__, 'architecture': platform.machine(), 'python': platform.python_version(), 'built_utc': stamp}, indent=2) + '\n')
    run([sys.executable, ROOT / 'scripts/check_versions.py', '--distribution', products])
    latest = ROOT / 'build/latest'
    temporary = ROOT / 'build/.latest-new'
    temporary.symlink_to(products.relative_to(latest.parent))
    temporary.replace(latest)
    print('Build products:', products)
    print('Install locally with ./scripts/install-local.sh')


if __name__ == '__main__':
    main()
