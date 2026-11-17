#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
kmz_setup_python
cd "$KMZ_ROOT"
.venv/bin/python scripts/check_versions.py
.venv/bin/python scripts/render_todo.py --check
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR="$KMZ_LOCAL_STATE_ROOT/matplotlib"
export KMZCONVERT_FIXTURES="$KMZ_ROOT/TestData/Local"
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp "$KMZ_ROOT/.local/pytest" "$@"
swift test --disable-sandbox --cache-path "$KMZ_ROOT/.cache/swiftpm" --config-path "$KMZ_ROOT/.local/swiftpm" --security-path "$KMZ_ROOT/.local/swiftpm-security"
