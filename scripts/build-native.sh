#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Pass --offline to use an already populated Cargo cache.
export CARGO_TARGET_DIR="$ROOT/artifacts/spine-target"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cargo build --manifest-path rust-spine/Cargo.toml --locked -p spine-cli "$@"
cargo build --manifest-path rust-spine/Cargo.toml --locked -p spine-cli --example unicode_nli_probe "$@"
cargo build --manifest-path spine-harness/Cargo.toml --locked --bins "$@"
