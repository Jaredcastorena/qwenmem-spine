#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${TRIANGLE_CPU_THREADS:-4}"
# Needs tokenizer + embedding assets and scripts/build-native.sh; no 4B inference.
.venv/bin/python scripts/make_retention_corpus.py
.venv/bin/python scripts/make_retention_boundaries.py
artifacts/spine-target/debug/qwenmem-spine-retention artifacts/retention/corpus.json artifacts/retention/native-v2 models/minilm
artifacts/spine-target/debug/qwenmem-spine-retention artifacts/retention/boundary-corpus.json artifacts/retention/native-boundaries models/minilm
artifacts/spine-target/debug/geometry > artifacts/retention/native-geometry.json
