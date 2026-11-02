"""Bounded local model runner. Never relax limits or restart a stopped job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 2**30
RESERVE = 16 * GIB
MAX_RAM = 24 * GIB


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot measure host RAM headroom')


def checked_cgroup():
    entry = next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    root = Path('/sys/fs/cgroup') / entry[3:].lstrip('/')
    limit = (root / 'memory.max').read_text().strip()
    if limit == 'max' or int(limit) > MAX_RAM:
        raise RuntimeError('Refusing unbounded model execution: 24 GiB cgroup RAM limit required')
    if (root / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Refusing model execution without disabled cgroup swap')
    return root


def stop_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def supervise(command, *, sample=available_bytes, reserve=RESERVE, interval=0.25):
    minimum = sample()
    if minimum < reserve:
        return dict(returncode=125, minimum_available_bytes=minimum,
                    stop_reason='host RAM headroom below reserve; test not started')
    child = subprocess.Popen(command, start_new_session=True)
    reason = None
    try:
        while child.poll() is None:
            free = sample()
            minimum = min(minimum, free)
            if free < reserve:
                reason = 'host RAM headroom below reserve; stopped test'
                stop_group(child)
                break
            time.sleep(interval)
    finally:
        if child.poll() is None:
            stop_group(child)
    return dict(returncode=125 if reason else child.returncode,
                minimum_available_bytes=minimum, stop_reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    directory = Path(__file__).resolve().parents[1] / 'artifacts/resource-guard'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'model.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another guarded model test is active')
        group = checked_cgroup()
        initial = available_bytes()
        if initial < MAX_RAM + RESERVE:
            raise SystemExit('Need 40 GiB available RAM before starting this bounded model test')
        stamp = time.time_ns()
        report = dict(command=command, initial_available_bytes=initial,
                      memory_max_bytes=int((group/'memory.max').read_text()),
                      memory_high_bytes=int((group/'memory.high').read_text()),
                      swap_max_bytes=0, reserve_bytes=RESERVE)
        target = directory / f'{stamp}.json'
        target.write_text(json.dumps(report, indent=2)+'\n')
        print('RAM guard: 24 GiB hard limit, 20 GiB throttle, no swap, 16 GiB host reserve.', flush=True)
        report.update(supervise(command))
        report['cgroup_peak_bytes'] = int((group/'memory.peak').read_text())
        report['memory_events'] = (group/'memory.events').read_text()
        target.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return report['returncode']


if __name__ == '__main__':
    sys.exit(main())
