#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$KMZ_ROOT"
.venv/bin/python scripts/install_local.py "$@"
