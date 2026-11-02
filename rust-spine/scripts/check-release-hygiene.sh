#!/usr/bin/env bash
set -euo pipefail

# A missing scanner or an unreadable file must never count as a clean release.
# Use grep instead of assuming ripgrep is installed on every release runner.
for required_command in grep mktemp rm python3; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    printf 'release hygiene requires %s\n' "$required_command" >&2
    exit 127
  fi
done

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

mode=${1:-git}
if (( $# > 1 )) || [[ "$mode" != git && "$mode" != --source-tree ]]; then
  echo "usage: scripts/check-release-hygiene.sh [--source-tree]" >&2
  exit 2
fi

failed=0
release_files=()
file_list=$(mktemp)
trap 'rm -f -- "$file_list"' EXIT

# Never let a Git-free source copy silently enumerate its parent repository.
# Source-tree mode is for an unpacked export; it includes hidden source files,
# excludes only build/interpreter caches, and rejects symbolic links.
if [[ "$mode" == --source-tree ]]; then
  python3 - "$file_list" <<'PY'
import os
import sys
from pathlib import Path

with open(sys.argv[1], "wb") as output:
    for directory, subdirs, files in os.walk(".", followlinks=False):
        for name in subdirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise SystemExit(f"release source contains a symbolic link: {path}")
        subdirs[:] = sorted(name for name in subdirs if name not in {
            ".git", "target", ".venv", "__pycache__", ".pytest_cache"
        })
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix not in {".pyc", ".pyo"}:
                output.write(os.fsencode(str(path)) + b"\0")
PY
else
  git_root=$(git rev-parse --show-toplevel)
  if [[ "$git_root" != "$repo_root" ]]; then
    echo "release source is not its own Git root; use --source-tree for an unpacked export" >&2
    exit 2
  fi
  # Keep this outside process substitution so a failed enumeration is fatal.
  git ls-files -z --cached --others --exclude-standard > "$file_list"
fi
while IFS= read -r -d '' file; do
  case "$file" in
    Cargo.lock|fuzz/Cargo.lock|scripts/check-release-hygiene.sh) continue ;;
  esac
  release_files+=("$file")
done < "$file_list"

if (( ${#release_files[@]} == 0 )); then
  echo "release hygiene found no candidate files to scan" >&2
  exit 1
fi

check_files() {
  local label="$1"
  local pattern="$2"
  local matches
  local status=0
  matches=$(LC_ALL=C grep -E -l -e "$pattern" -- "${release_files[@]}") || status=$?
  case "$status" in
    0)
      echo "$label found in:"
      printf '%s\n' "$matches"
      failed=1
      ;;
    1) ;; # grep uses 1 for a successful scan with no matches.
    *)
      printf 'release hygiene scanner failed for %s (exit %s)\n' "$label" "$status" >&2
      exit "$status"
      ;;
  esac
}

check_files "absolute developer home path" '(/h[o]me/[^/[:space:]]+|/U[s]ers/[^/[:space:]]+|[A-Za-z]:\\U[s]ers\\[^\\[:space:]]+)'
check_files "private or tailnet IPv4 literal" '(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3})'
check_files "credential-shaped value" '(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|PRIVATE[[:space:]]KEY-----)'
check_files "credential embedded in URL" 'https?://[^/@[:space:]]+:[^/@[:space:]]+@'

if (( failed != 0 )); then
  exit 1
fi

python3 scripts/check-vendored-candle.py

echo "release hygiene checks passed"
