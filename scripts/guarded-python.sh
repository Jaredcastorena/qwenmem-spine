#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${INTEL_RUNTIME_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="${INTEL_RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu:${INTEL_RUNTIME_ROOT}/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export OCL_ICD_FILENAMES="${INTEL_RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
fi
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${TRIANGLE_CPU_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$ROOT"
# Prefer the highest CPPC-ranked physical cores, using only one SMT sibling each.
CPU_LIST="$("$ROOT/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
cores={}
for n in os.sched_getaffinity(0):
 p=Path(f'/sys/devices/system/cpu/cpu{n}')
 try:
  key=((p/'topology/physical_package_id').read_text(),(p/'topology/core_id').read_text())
  rank=int((p/'acpi_cppc/highest_perf').read_text())
 except OSError:
  key=(0,n); rank=0
 if key not in cores or n<cores[key][1]: cores[key]=(rank,n)
ordered=sorted(cores.values(),key=lambda x:(-x[0],x[1]))
print(','.join(str(n) for _,n in ordered[:int(os.environ['OMP_NUM_THREADS'])]))
PY
)"
# Fail closed if this host cannot enforce the test's RAM budget. The watchdog
# additionally monitors global headroom, including memory used outside this scope.
exec systemd-run --user --scope --quiet \
  --property=MemoryHigh=20G --property=MemoryMax=24G \
  --property=MemorySwapMax=0 \
  /usr/bin/python3 "$ROOT/scripts/resource_guard.py" -- \
  taskset -c "$CPU_LIST" "$ROOT/.venv/bin/python" "$@"

