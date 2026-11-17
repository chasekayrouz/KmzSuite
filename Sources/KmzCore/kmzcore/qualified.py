"""Dependency versions that the exact performance fast paths were qualified against.

Several speedups equal the original code only because they reproduce, or reach
into, a pinned dependency: Qt's raster engine and QSortFilterProxyModel,
pyqtgraph's paint internals, pyteomics' mzML index and arrays, NumPy's semantics
and ms-entropy's index. Each engages only where ``qualified`` confirms that the
installed versions equal these ``requirements-build.lock`` pins. Any other
version takes the original path: slower, never different. Before changing a
pin, add the new version here and rerun the parity tests, the converter
byte-identity check and a GUI digest/pixel comparison.
"""
from __future__ import annotations

from functools import cache
import importlib

QUALIFIED = {
    "Qt": "6.11.2",  # The Qt libraries PySide6 loads (qVersion); PySide6 wheels ship their own.
    "PySide6": "6.11.2",
    "pyqtgraph": "0.14.0",
    "numpy": "2.5.3",
    "pyteomics": "5.0.1",
    "ms-entropy": "1.5.3",
}
# Read from __version__: callers check these only when their fast path has already
# imported the package, and frozen apps need no copied metadata for them.
_MODULES = {"PySide6": "PySide6", "pyqtgraph": "pyqtgraph", "numpy": "numpy", "ms-entropy": "ms_entropy"}


def installed(name: str) -> str | None:
    """The installed version of a dependency in QUALIFIED, or None if unreadable."""
    try:
        if name == "Qt":
            from PySide6.QtCore import qVersion
            return qVersion()
        if name == "pyteomics":
            # Importing pyteomics takes ~0.2 s; the frozen viewers and CoMet copy its metadata.
            from importlib.metadata import version
            return version(name)
        return importlib.import_module(_MODULES[name]).__version__
    except (ImportError, AttributeError):
        return None


@cache
def qualified(*names: str) -> bool:
    """True when every named dependency is installed at its QUALIFIED version."""
    return all(installed(name) == QUALIFIED[name] for name in names)
