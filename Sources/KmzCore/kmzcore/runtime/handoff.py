"""Start an independent MS2 session using the current installation/runtime."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys


def precursor_mass(value: str) -> float:
    mass = float(value)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Precursor m/z must be finite and positive.')
    return mass


def ms2_command(paths=(), *, precursor_mz=None) -> list[str]:
    arguments = []
    if precursor_mz is not None:
        arguments = ['--precursor-mz', repr(precursor_mass(str(precursor_mz)))]
    arguments += ['--', *(str(Path(path).resolve()) for path in paths)]
    if getattr(sys, 'frozen', False):
        source = Path(sys.executable).resolve().parents[2]
        sibling = source.parent / 'Kmz2Viewer.app'
        if not (sibling / 'Contents/MacOS/Kmz2Viewer').is_file():
            raise FileNotFoundError('Kmz2Viewer.app must be beside KmzViewer.app. '
                                    'Keep the KmzSuite applications together.')
        return ['/usr/bin/open', '-n', '-a', str(sibling), '--args', *arguments]
    return [sys.executable, '-m', 'kmz2viewer', *arguments]


def open_ms2(paths=(), *, precursor_mz=None) -> None:
    command = ms2_command(paths, precursor_mz=precursor_mz)
    if getattr(sys, 'frozen', False):
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise OSError(result.stderr.strip() or 'macOS could not launch Kmz2Viewer.')
    else:
        from PySide6.QtCore import QProcess
        started, _pid = QProcess.startDetached(command[0], command[1:])
        if not started:
            raise OSError('Could not start Kmz2Viewer with the current Python runtime.')
