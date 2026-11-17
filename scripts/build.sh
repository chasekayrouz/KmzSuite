#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
kmz_setup_python
cd "$KMZ_ROOT"
.venv/bin/python scripts/check_versions.py
.venv/bin/python scripts/render_todo.py --check
swift build -c release --disable-sandbox --cache-path "$KMZ_ROOT/.cache/swiftpm" --config-path "$KMZ_ROOT/.local/swiftpm" --security-path "$KMZ_ROOT/.local/swiftpm-security"
.venv/bin/python scripts/build.py
