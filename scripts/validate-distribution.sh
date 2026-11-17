#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$KMZ_ROOT"
KMZ_DISTRIBUTION="${1:-$HOME/Applications/KmzSuite}"
.venv/bin/python scripts/check_versions.py --distribution "$KMZ_DISTRIBUTION"
.venv/bin/python scripts/render_todo.py --check
.venv/bin/python scripts/smoke-viewers.py "$KMZ_DISTRIBUTION" --in-place --report "$KMZ_ROOT/.local/validation/viewer-receipt-$(date -u +%Y%m%dT%H%M%S).json"
.venv/bin/python scripts/smoke-native.py --distribution "$KMZ_DISTRIBUTION"
