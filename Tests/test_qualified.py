"""Exact fast paths engage only on the dependency versions pinned for distribution builds."""
from pathlib import Path
import subprocess
import sys

from kmzcore import qualified as module
from kmzcore.qualified import QUALIFIED, installed, qualified

ROOT = Path(__file__).resolve().parents[1]


def test_qualified_versions_are_the_build_pins():
    """Bumping a pin fails here until its fast paths are requalified and this table updated."""
    pins = dict(line.split('==') for line in (ROOT / 'requirements-build.lock').read_text().splitlines()
                if '==' in line)
    assert QUALIFIED['Qt'] == pins['PySide6']  # PySide6 wheels ship the Qt they wrap.
    assert {name: QUALIFIED[name] for name in QUALIFIED if name != 'Qt'} == {
        name: pins[name] for name in QUALIFIED if name != 'Qt'}


def test_development_environment_runs_the_fast_paths():
    assert {name: installed(name) for name in QUALIFIED} == QUALIFIED
    assert qualified(*QUALIFIED)


def test_unreadable_or_other_versions_are_unqualified(monkeypatch, unqualified):
    monkeypatch.setitem(module._MODULES, 'pyqtgraph', 'kmzsuite_missing_module')
    unqualified()  # clears the cache
    assert installed('pyqtgraph') is None and not qualified('pyqtgraph')
    assert qualified('numpy')
    unqualified('numpy')
    assert not qualified('numpy') and not qualified('Qt', 'numpy') and qualified('Qt')


def test_versions_read_without_importing_heavy_packages():
    code = ('import sys\nfrom kmzcore.qualified import qualified\n'
            'assert qualified("pyteomics", "numpy")\n'
            'print(sorted(m for m in ("pyteomics", "psims", "matplotlib", "openpyxl", "PySide6", "pyqtgraph")'
            ' if m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == '[]'
