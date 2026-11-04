"""Exercise acceptance exit status with synthetic files and no model process."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'spine_end_to_end', Path(__file__).resolve().parents[1] / 'scripts/spine_end_to_end.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
CHECKPOINT = 'c' * 64


@pytest.fixture
def acceptance(tmp_path, monkeypatch):
    root = tmp_path / 'fixtures'

    def write(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    corpus = {'queries': [{'id': 'beacon', 'question': 'What is the beacon?',
                           'expected': 'BEACON-123', 'support': ['beacon']}]}
    native = {'native': {'canonical_events': 1, 'event_labels': [
        {'label': 'beacon', 'event_id': 'event:beacon'}]}}
    for folder, metadata in [('native-v2', '128k.json'), ('native-boundaries', 'boundary.json')]:
        write(f'artifacts/retention/{folder}/{metadata}', native)
        write(f'artifacts/retention/{folder}/retention.spine', {'synthetic': True})
    for filename in ['corpus.json', 'boundary-corpus.json']:
        write(f'artifacts/retention/{filename}', corpus)
    write('models/qwen3.5-triangles/triangle.json', {'weights_sha256': CHECKPOINT})
    write('fake-spine', {'binary': 'unused'})
    monkeypatch.setattr(driver, 'ROOT', root)
    monkeypatch.setattr(driver, 'BINARY', root / 'fake-spine')
    monkeypatch.setattr(driver.os, 'environ', driver.os.environ.copy())

    class FakeApp:
        gate_verified = True
        branch = True
        error = None
        answer = 'BEACON-123 [event:beacon]'
        instances = []

        def __init__(self, out, heart, workspace, phase):
            self.log = out / f'{phase}.log'
            self.events = []
            self.closed = False
            self.instances.append(self)

        def ask(self, label, text):
            assert label == 'beacon', 'Shortened runs must skip operational checks'
            if self.error is not None:
                raise self.error
            execution = {'engine': 'merged_checkpoint_prefix', 'prefix_layers': 5,
                         'state_before': 'root', 'state_after': 'selected',
                         'ticks_before': 0, 'ticks_after': 1}
            self.events.append({
                'type': 'triangle_internal_bank', 'bank': 0, 'resident_records': 3,
                'resident_triangles': 1, 'commands': ['left'] if self.branch else [],
                'source_gate': 'trace_verified', 'forest_sha256': 'forest',
                'execution': [execution]})
            self.events.append({
                'type': 'triangle_chat', 'bank_count': 1, 'resident_records': 3,
                'resident_triangles': 1, 'branch_commands': int(self.branch),
                'forest_sha256': 'forest', 'retrieval': 'compiled_native_handoff',
                'source_gate_verified': self.gate_verified, 'weights_sha256': CHECKPOINT,
                'navigation_unchanged': True,
                'execution': dict(execution, state_before='selected', ticks_before=1)})
            return self.answer, {'phase': 'idle'}

        def close(self):
            assert not self.closed
            self.closed = True
            # The real reader flushes its last audit events before close returns.
            self.log.write_text(''.join(json.dumps(event) + '\n' for event in self.events))

    monkeypatch.setattr(driver, 'App', FakeApp)
    out = tmp_path / 'result'

    def run(*flags):
        monkeypatch.setattr(driver.sys, 'argv', ['spine_end_to_end.py', '--output', str(out), *flags])
        return driver.main()

    return run, out, FakeApp


@pytest.mark.parametrize('mode', [('--pilot',), ('--boundaries',), ('--query', 'beacon')])
@pytest.mark.parametrize('proof_passes', [False, True])
def test_shortened_run_exits_according_to_final_navigation_proof(acceptance, mode, proof_passes):
    run, out, app = acceptance
    app.gate_verified = proof_passes
    flags = (*mode, '--triangle-internal-context', '--require-triangle-branch')
    if proof_passes:
        assert run(*flags) is None  # Normal return from the script entry point exits zero.
    else:
        with pytest.raises(SystemExit) as failure:
            run(*flags)
        assert failure.value.code not in (None, 0)
    report = json.loads((out / 'results.json').read_text())
    assert report['behavioral_all_passed'] is True
    assert report['internal_navigation']['passed'] is proof_passes
    assert report['internal_navigation']['completed_rounds_bound_to_banks'] is True
    assert report['all_passed'] is proof_passes
    assert 'run_error' not in report  # A failed verdict is distinct from an execution error.
    assert all(instance.closed for instance in app.instances)
    events = json.loads((out / 'internal-navigation-events.json').read_text())
    assert len(events['chats']) == len(events['banks']) == 1


def test_required_branch_failure_is_retained(acceptance):
    run, out, app = acceptance
    app.branch = False
    with pytest.raises(SystemExit):
        run('--pilot', '--triangle-internal-context', '--require-triangle-branch')
    report = json.loads((out / 'results.json').read_text())
    assert report['behavioral_all_passed'] is True
    assert report['internal_navigation']['all_completions_gated'] is True
    assert report['internal_navigation']['branch_exercised'] is False
    assert report['all_passed'] is False


def test_behavioral_failure_still_exits_nonzero(acceptance):
    run, out, app = acceptance
    app.answer = 'Wrong beacon [event:beacon]'
    with pytest.raises(SystemExit):
        run('--pilot', '--triangle-internal-context')
    report = json.loads((out / 'results.json').read_text())
    assert report['behavioral_all_passed'] is False
    assert report['internal_navigation']['passed'] is True
    assert report['all_passed'] is False


def test_execution_error_keeps_original_exception_and_report(acceptance):
    run, out, app = acceptance
    app.error = RuntimeError('synthetic turn failure')
    with pytest.raises(RuntimeError) as failure:
        run('--pilot', '--triangle-internal-context')
    assert failure.value is app.error
    report = json.loads((out / 'results.json').read_text())
    assert report['run_error'] == 'synthetic turn failure'
    assert report['run_completed'] is False
    assert report['all_passed'] is False
    assert app.instances[0].closed is True


def test_branch_requirement_without_internal_context_is_rejected_before_start(acceptance, capsys):
    run, out, app = acceptance
    with pytest.raises(SystemExit) as failure:
        run('--pilot', '--require-triangle-branch')
    assert failure.value.code == 2
    assert '--require-triangle-branch requires --triangle-internal-context' in capsys.readouterr().err
    assert not out.exists()
    assert not app.instances
