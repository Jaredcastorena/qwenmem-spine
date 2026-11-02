#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$LAB_ROOT/scripts/guarded-python.sh" "$LAB_ROOT/scripts/spine_local.py" "$@"
