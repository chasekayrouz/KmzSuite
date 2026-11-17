"""Validate a staged distribution, then atomically install durable user apps."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile
from macos_apps import APPS, atomic_publish, require_stopped, validate_bundle
from check_versions import check_distribution
from installed_apps import archive_rollback, installed_directory, isolated_smoke, register

ROOT = Path(__file__).resolve().parents[1]


def install(source, current, output):
    source, current = Path(source).resolve(strict=True), Path(current).absolute()
    if current.is_symlink() or current.parent.is_symlink():
        raise RuntimeError('Installation paths must be real directories')
    if current.resolve().is_relative_to(ROOT):
        raise RuntimeError('Production apps must be outside the checkout')
    if not any(current.resolve().is_relative_to(base) for base in (Path.home() / 'Applications', Path('/Applications'))):
        raise RuntimeError('Choose a durable location under an Applications directory')
    require_stopped(current)
    if current.exists() and not (current / '.kmz-install.json').is_file():
        raise RuntimeError('Refusing to replace an unmanaged application directory')
    allowed = set(APPS) | {'Converter', 'kmzconvert', 'licenses', 'LICENSE.txt', 'THIRD_PARTY_NOTICES.txt',
                           'build.json', '.kmz-install.json', '.DS_Store'}
    if current.exists() and any(p.name not in allowed for p in current.iterdir()):
        raise RuntimeError('Installed directory contains additional user files; preserve them before updating')
    rollback = current.parent / '.KmzSuite-rollback.zip'
    if rollback.is_symlink() or (rollback.exists() and not rollback.is_file()):
        raise RuntimeError('Rollback must be a regular archive')
    if rollback.exists():
        with ZipFile(rollback) as zipped:
            owners = [n for n in zipped.namelist() if n.endswith('/.kmz-install.json')]
            if len(owners) != 1 or not json.loads(zipped.read(owners[0])).get('ok'):
                raise RuntimeError('Refusing to replace an unmanaged rollback archive')
    for name in APPS:
        validate_bundle(source / name)
    check_distribution(source)
    current.parent.mkdir(parents=True, exist_ok=True)
    stage = current.parent / ('.KmzSuite-install-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    subprocess.run(['/usr/bin/ditto', '--rsrc', '--extattr', '--acl', str(source), str(stage)], check=True)
    promoted = False
    had_current = current.exists()
    rollback_stage = current.parent / (stage.name + '-rollback.zip')
    try:
        check_distribution(stage)
        isolation_output = ROOT.parent / ('.KmzSuite-install-verification-' + stage.name.rsplit('-', 1)[-1])
        receipt = isolated_smoke(stage, ROOT, isolation_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(isolation_output), str(output))
        (stage / '.kmz-install.json').write_text(json.dumps(receipt, indent=2) + '\n')
        require_stopped(current)
        if had_current:
            archive_rollback(current, rollback_stage, output / 'rollback-roundtrip')
        atomic_publish(stage, current)
        promoted = True
        register(current)
        if had_current:
            os.replace(rollback_stage, rollback)
    except BaseException:
        if promoted and had_current:
            require_stopped(current)
            atomic_publish(stage, current)
            register(current)
        if stage.exists():
            require_stopped(stage)
            shutil.rmtree(stage)
        if had_current and not promoted:
            register(current)  # Candidate smoke may have registered a stage path.
        rollback_stage.unlink(missing_ok=True)
        raise
    if had_current:
        if stage.exists():
            try:
                shutil.rmtree(stage)  # Only the superseded, managed rollback.
            except OSError:
                print('Old rollback cleanup failed; usable installation is intact. Inspect:', stage)
        print('One previous working distribution retained:', rollback)
    print('Installed:', current)
    print('Launch KmzSuite, KmzViewer, Kmz2Viewer, KmzConvert or KmzCoMet with Cmd+Space.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'build/latest')
    parser.add_argument('--destination', type=Path, default=installed_directory())
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    install(args.source, args.destination, ROOT / '.local/validation/install' / stamp)


if __name__ == '__main__':
    main()
