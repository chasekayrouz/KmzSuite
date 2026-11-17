#!/bin/bash
set -euo pipefail
KMZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$KMZ_ROOT/.cache/uv"
# Builds use uv-managed CPython for the series in .python-version, downloaded on
# first use; system and Homebrew interpreters are ignored.
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_DOWNLOADS=automatic
export PIP_CACHE_DIR="$KMZ_ROOT/.cache/pip"
export TMPDIR="$KMZ_ROOT/.local/tmp"
mkdir -p "$TMPDIR" "$KMZ_ROOT/.local/state"
export KMZ_LOCAL_STATE_ROOT="$KMZ_ROOT/.local/state"
export PYTHONDONTWRITEBYTECODE=1
export PYINSTALLER_CONFIG_DIR="$KMZ_ROOT/.cache/pyinstaller"
export CLANG_MODULE_CACHE_PATH="$KMZ_ROOT/.build/clang-module-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$KMZ_ROOT/.build/swift-module-cache"

kmz_setup_python() {
    command -v uv >/dev/null || { echo "Install uv before building (see README)." >&2; return 1; }
    local kmz_python_version
    kmz_python_version="$(cat "$KMZ_ROOT/.python-version")"
    if [[ ! -x "$KMZ_ROOT/.venv/bin/python" ]]; then
        local kmz_interpreter
        kmz_interpreter="${KMZ_PYTHON:-$kmz_python_version}"
        uv venv --python "$kmz_interpreter" "$KMZ_ROOT/.venv"
    fi
    "$KMZ_ROOT/.venv/bin/python" - "$kmz_python_version" <<'PY'
import platform
import sys
series = sys.argv[1].split('.')
found = platform.python_version()
if platform.python_implementation() != 'CPython' or found.split('.')[:len(series)] != series:
    raise SystemExit(f'KmzSuite requires CPython {".".join(series)}; this .venv uses '
                     f'{platform.python_implementation()} {found}. Delete .venv and run the '
                     'build again to recreate it.')
PY
    local kmz_lock_hash
    kmz_lock_hash="$(shasum -a 256 "$KMZ_ROOT/requirements-build.lock" | cut -d ' ' -f 1)"
    if [[ ! -f "$KMZ_ROOT/.venv/.kmz-lock" ]] || [[ "$(cat "$KMZ_ROOT/.venv/.kmz-lock")" != "$kmz_lock_hash" ]]; then
        uv pip sync --python "$KMZ_ROOT/.venv/bin/python" "$KMZ_ROOT/requirements-build.lock"
        printf '%s\n' "$kmz_lock_hash" > "$KMZ_ROOT/.venv/.kmz-lock"
    fi
    uv pip install --python "$KMZ_ROOT/.venv/bin/python" --no-build-isolation --no-deps -e "$KMZ_ROOT"
}
