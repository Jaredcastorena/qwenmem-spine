import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('resource_guard', Path(__file__).resolve().parents[1]/'scripts/resource_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_headroom_drop_stops_child_without_allocating_ram():
    readings = iter([100, 100, 9])
    result = guard.supervise([sys.executable, '-c', 'import time; time.sleep(60)'],
                             sample=lambda: next(readings, 9), reserve=10, interval=0.01)
    assert result['returncode'] == 125
    assert result['minimum_available_bytes'] == 9
    assert result['stop_reason']


def test_successful_child_status_is_preserved():
    result = guard.supervise([sys.executable, '-c', 'raise SystemExit(7)'],
                             sample=lambda: 100, reserve=10, interval=0.01)
    assert result['returncode'] == 7
    assert result['stop_reason'] is None
