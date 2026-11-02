#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [[ "$#" -eq 0 ]]; then
  set -- demo
fi
# The default standalone circuit is small and runs on CPU. For merged XPU runs,
# use scripts/guarded-python.sh -m triangle_memory.cli --device xpu --model ... .
exec "$ROOT/.venv/bin/python" -m triangle_memory.cli "$@"
