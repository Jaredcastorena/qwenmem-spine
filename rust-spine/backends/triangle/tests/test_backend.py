import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triangle_backend.backend import Backend, serve
from triangle_backend.forest import Forest
from triangle_backend import limits


def sample_forest():
    return {"schema": 1, "nodes": {name: {"text": f"Source {name}", "event_id": f"event:{name}",
        "vector": [1.0, 0.0]} for name in ("a", "b", "c")}, "triangles": {
        "d": {"apex": "a", "left": "a", "right": "b"},
        "e": {"apex": "a", "left": "d", "right": "c"}}, "roots": ["e"]}


class FakeRunner:
    weight_id = "test-weights"

    def __init__(self, forest, **kwargs):
        self.forest = forest
        self.calls = []
        self.replacements = 0

    def replace_forest(self, forest):
        self.replacements += 1
        self.forest = forest
        self.calls.clear()

    def execute(self, token):
        self.calls.append(token)
        return [token, 267]

    def save(self, path):
        self.calls.append(("save", path))

    def restore(self, path):
        self.calls.append(("restore", path))

    def answer(self, question, evidence, max_tokens):
        self.last_evidence = evidence
        return {"answer": "Supported event:a. Unrelated event:absent and event:ab. Node a.",
            "prompt_tokens": 12, "answer_tokens": 8, "truncated": False}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend(model_dir="/absent", compiler_dir="/absent", runner_factory=FakeRunner)
        self.counter = 0

    def call(self, op, **args):
        self.counter += 1
        return self.backend.handle({"schema": 1, "id": self.counter, "op": op, "args": args})

    def install(self, forest=None):
        result = self.call("install", forest=sample_forest() if forest is None else forest)
        self.assertTrue(result["ok"], result)
        return result["result"]["forest_sha256"]

    def test_chat_contract_checks_revision_budget_tools_and_temperature_before_loading(self):
        revision = self.install()
        args = dict(forest_sha256=revision, messages=[{"role":"user","content":"Q"}],
            tools=[], max_tokens=100, temperature=0.0)
        for change in ({"forest_sha256":"old"}, {"max_tokens":2049},
                {"messages":[{"role":"invalid","content":"Q"}]},
                {"temperature":float("nan")}, {"tools":[{"type":"function","function":{}}]}):
            self.assertFalse(self.call("complete", **(args | change))["ok"])
            self.assertIsNone(self.backend.runner)

    def test_lazy_handshake_install_and_shutdown(self):
        result = self.call("hello")["result"]
        self.assertEqual((result["protocol"], result["capacity"], result["max_depth"]), (1, 64, 8))
        self.install()
        self.assertIsNone(self.backend.runner)
        self.assertEqual(self.backend.forest.addresses, ["a", "b", "c", "d", "e"])
        self.assertTrue(self.call("shutdown")["ok"])
        self.assertFalse(self.call("hello")["ok"])

    def test_revision_before_mutation_and_identical_install_retains_state(self):
        revision = self.install()
        self.assertTrue(self.call("execute", forest_sha256=revision, command="open:4")["ok"])
        runner = self.backend.runner
        self.assertEqual(runner.calls, [4])
        for op, args in (("execute", {"command": "left"}), ("save", {"path": "state"}),
                         ("restore", {"path": "state"})):
            result = self.call(op, forest_sha256="old", **args)
            self.assertEqual(result["error"]["code"], "forest_revision")
        self.assertEqual(runner.calls, [4])
        self.assertFalse(self.call("install", forest=sample_forest())["result"]["session_reset"])
        changed = sample_forest()
        changed["nodes"]["a"]["text"] = "new content"
        new_revision = self.install(changed)
        self.assertNotEqual(new_revision, revision)
        self.assertEqual(runner.replacements, 1)
        self.assertEqual(runner.calls, [])

    def test_invalid_install_preserves_forest_and_session(self):
        revision = self.install()
        self.call("execute", forest_sha256=revision, command="peek")
        bad = sample_forest()
        bad["triangles"]["d"]["left"] = "e"
        result = self.call("install", forest=bad)
        self.assertEqual(result["error"]["code"], "invalid_forest")
        self.assertEqual(self.backend.forest.digest, revision)
        self.assertEqual(self.backend.runner.calls, [264])

    def test_empty_forest_can_generate_but_not_navigate(self):
        revision = self.install({"schema": 1, "nodes": {}, "triangles": {}, "roots": []})
        result = self.call("execute", forest_sha256=revision, command="peek")
        self.assertEqual(result["error"]["code"], "empty_forest")
        self.assertIsNone(self.backend.runner)
        self.assertTrue(self.call("generate", question="What is known?", evidence=[], max_tokens=12)["ok"])

    def test_generation_preserves_validated_source_chronology_and_spans(self):
        self.install()
        item = {"node_id": "a", "event_id": "event:a", "text": "Correction",
            "canonical_index": 4, "device_id": "device:one", "device_sequence": 8,
            "start_char": 1200, "end_char": 1210, "source_chars": 1400, "role": "User"}
        self.assertTrue(self.call("generate", question="Current value?", evidence=[item], max_tokens=12)["ok"])
        self.assertEqual(self.backend.runner.last_evidence, [item])
        for value in (True, -1, 1.5, 2**64):
            malformed = {**item, "device_sequence": value}
            self.assertFalse(self.call("generate", question="Current value?", evidence=[malformed], max_tokens=12)["ok"])

    def test_generation_bounds_citations_and_navigation_preserved(self):
        revision = self.install()
        self.call("execute", forest_sha256=revision, command="open:4")
        evidence = [{"node_id": "a", "event_id": "event:a", "text": "Fact", "omission": "tail"}]
        result = self.call("generate", question="Fact?", evidence=evidence, max_tokens=12)
        self.assertEqual(result["result"]["citations"], ["event:a", "a"])
        self.assertEqual(self.backend.runner.calls, [4])
        for budget in (0, 513, True, 1.5):
            self.assertFalse(self.call("generate", question="Q", evidence=evidence, max_tokens=budget)["ok"])
        self.assertFalse(self.call("generate", question="x"*(limits.MAX_QUESTION_BYTES+1), evidence=[], max_tokens=1)["ok"])
        self.assertFalse(self.call("generate", question="Q", evidence=evidence*65, max_tokens=1)["ok"])
        self.assertFalse(self.call("generate", question="Q", evidence=[{}], max_tokens=1)["ok"])

    def test_invalid_request_and_command_types(self):
        revision = self.install()
        for invalid in (-1, 2**64, True, 1.5, "1"):
            response = self.backend.handle({"schema": 1, "id": invalid, "op": "hello", "args": {}})
            self.assertFalse(response["ok"])
            self.assertEqual(response["id"], 0)
        for command in ("open:64", "route:8", "run", "open:-1", True):
            self.assertFalse(self.call("execute", forest_sha256=revision, command=command)["ok"])
        self.assertIsNone(self.backend.runner)
        self.assertFalse(self.call("hello", extra=1)["ok"])

    def test_unexpected_model_failure_closes_session(self):
        revision = self.install()
        self.call("execute", forest_sha256=revision, command="peek")
        def fail(_):
            raise RuntimeError("Nonexact circuit output")
        self.backend.runner.execute = fail
        self.assertEqual(self.call("execute", forest_sha256=revision, command="left")["error"]["code"], "backend_failure")
        self.assertTrue(self.backend.closed)
        self.assertEqual(self.call("save", forest_sha256=revision, path="unused")["error"]["code"], "closed")

    def test_stream_rejects_nonfinite_duplicate_oversized_and_partial_json(self):
        for line in (b'{"schema":1,"id":1,"id":2,"op":"hello","args":{}}\n',
                     b'{"value":NaN}\n', b'\xff\n', b'{}', b'x'*(limits.MAX_LINE_BYTES+1)):
            output = io.BytesIO()
            code = serve(self.backend, io.BytesIO(line), output)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["error"]["code"], "invalid_json")

    def test_cli_has_no_model_or_compiler_dependency_until_needed(self):
        requests = b'{"schema":1,"id":1,"op":"hello","args":{}}\n{"schema":1,"id":2,"op":"shutdown","args":{}}\n'
        result = subprocess.run([sys.executable, str(ROOT/"server.py"), "--model-dir", "/absent",
            "--compiler-dir", "/absent"], input=requests, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=True)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["id"] for r in responses], [1, 2])
        self.assertTrue(all(r["ok"] for r in responses))


class ForestTests(unittest.TestCase):
    def test_digest_is_content_and_order_stable_with_detached_input(self):
        data = sample_forest()
        forest = Forest(data)
        shuffled = copy.deepcopy(data)
        shuffled["nodes"] = dict(reversed(list(shuffled["nodes"].items())))
        self.assertEqual(forest.digest, Forest(shuffled).digest)
        data["nodes"]["a"]["text"] = "modified externally"
        self.assertNotEqual(forest.data["nodes"]["a"]["text"], data["nodes"]["a"]["text"])

    def test_capacity_vector_roots_and_overlap_validation(self):
        mutations = [lambda d: d["nodes"]["a"].update(vector=[0, 0]),
            lambda d: d["nodes"]["a"].update(vector=[float("nan")]),
            lambda d: d["nodes"]["a"].update(vector=[True]),
            lambda d: d.update(roots=["missing"]), lambda d: d.update(roots=[]),
            lambda d: d["triangles"].update(a=d["triangles"]["d"]),
            lambda d: d["nodes"].update({f"extra{i}": copy.deepcopy(d["nodes"]["a"]) for i in range(64)})]
        for mutate in mutations:
            data = sample_forest()
            mutate(data)
            with self.assertRaises((ValueError, TypeError)):
                Forest(data)

    def test_finite_parallel_vectors_avoid_intermediate_norm_overflow(self):
        for magnitude in (1e100, 1e-100):
            data = sample_forest()
            for node in data["nodes"].values():
                node["vector"] = [magnitude]
            self.assertEqual(len(Forest(data).addresses), 5)


if __name__ == "__main__":
    unittest.main()
