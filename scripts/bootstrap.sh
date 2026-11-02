#!/usr/bin/env bash
# Install Python dependencies only; model downloads and compilation are explicit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE="${1:-cpu}"
if [[ "$#" -gt 1 || ( "$DEVICE" != cpu && "$DEVICE" != xpu ) ]]; then
  echo "Usage: scripts/bootstrap.sh [cpu|xpu]" >&2
  exit 2
fi
python3 -m venv .venv
# The plain version pin also matches +cpu/+xpu; reinstall when selecting a device.
.venv/bin/python -m pip install --force-reinstall "torch==2.14.1" --index-url "https://download.pytorch.org/whl/$DEVICE"
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e '.[test]'
echo "Next: .venv/bin/python scripts/download.py --compiler-only"
