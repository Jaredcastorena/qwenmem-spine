"""Optional CPU-only checks using the caller's existing runtime/compiler environment."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triangle_backend.forest import Forest
from triangle_backend.session import CircuitSession
from test_backend import sample_forest

COMPILER = os.environ.get("TRIANGLE_COMPILER_DIR")


@unittest.skipUnless(COMPILER, "Set TRIANGLE_COMPILER_DIR for existing local compiler CPU checks")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from triangle_backend.upstream import configure
        configure(COMPILER)
        from triangle_backend import circuit
        import torch
        from chipformer.q35 import tiny
        from chipformer.rv.hf import inject
        from triangle_backend import protocol as p
        torch.set_num_threads(2)

        class TinySession(CircuitSession):
            def __init__(self):
                self.device = "cpu"
                self.forest = Forest(sample_forest())
                self.circuit = circuit.build()
                circuit.install(self.circuit, self.forest)
                self.model = tiny.build(self.circuit, list(range(p.VOCAB_SIZE)),
                    [next(iter(e.terms)) for e in self.circuit.code], dtype=torch.bfloat16)
                self.model.config._attn_implementation = "eager"
                self.model.eval()
                self.weight_id = "test-only-tiny-circuit"
                self.reset()

            def reset(self):
                with torch.no_grad():
                    self.cache = self.model(torch.tensor([[p.VOCAB_SIZE]]), use_cache=True).past_key_values
                    inject(self.circuit, self.cache)

            def tick(self, token):
                with torch.no_grad():
                    result = self.model(torch.tensor([[token]]), past_key_values=self.cache, use_cache=True)
                    self.cache = result.past_key_values
                    logits = result.logits[0, -1, :p.VOCAB_SIZE]
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("Non-finite circuit logits")
                    return int(logits.argmax())

        cls.runner = TinySession()

    def setUp(self):
        self.runner.reset()

    def test_real_compiled_cpu_trace_private_save_restore_and_corruption(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        runner = self.runner
        initial_digest = runner.state_digest()
        self.assertEqual(runner.execute(4), [4, 192, 267])
        self.assertNotEqual(runner.state_digest(), initial_digest)
        self.assertEqual(runner.execute(265), [265, 130, 67, 192, 267])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.safetensors"
            runner.save(path)
            saved_digest = runner.state_digest()
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            runner.reset()
            runner.restore(path)
            self.assertEqual(runner.state_digest(), saved_digest)
            self.assertEqual(runner.execute(266), [266, 128, 65, 193, 267])
            self.assertEqual(runner.execute(256), [256, 66, 194, 267])
            with safe_open(path, framework="pt") as saved:
                metadata = saved.metadata()
                tensors = {key: saved.get_tensor(key) for key in saved.keys()}
            first = next(iter(tensors.values())).view(-1)
            first[0] = -first[0] if first[0] != 0 else 1
            save_file(tensors, path, metadata=metadata)
            before = runner._state()
            with self.assertRaisesRegex(ValueError, "damaged"):
                runner.restore(path)
            self.assertTrue(all(torch.equal(value, runner._state()[key]) for key, value in before.items()))

    def test_snapshot_identity_rejections_do_not_mutate_navigation(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.safetensors"
            self.runner.execute(4)
            self.runner.save(path)
            with safe_open(path, framework="pt") as saved:
                original = saved.metadata()
                tensors = {key: saved.get_tensor(key) for key in saved.keys()}
            before = self.runner._state()
            for field in ("schema", "protocol_sha256", "forest_sha256", "weights_sha256"):
                metadata = dict(original)
                metadata[field] = "wrong"
                save_file(tensors, path, metadata=metadata)
                with self.assertRaises(ValueError):
                    self.runner.restore(path)
                self.assertTrue(all(torch.equal(value, self.runner._state()[key]) for key, value in before.items()))

    def language_runner(self):
        from triangle_backend.merged import MergedRunner
        runner = MergedRunner.__new__(MergedRunner)
        runner.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=10000))
        runner.meta = {"text_vocab": 4}
        runner.cache = object()
        runner.tokenizer = SimpleNamespace(eos_token_id=3,
            apply_chat_template=lambda *args, **kwargs: [0, 1],
            decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
        return runner

    def test_generation_uses_fresh_cache_and_honors_eos_and_token_bound(self):
        import torch
        runner = self.language_runner()
        original = runner.cache
        caches = []
        def step(token, cache, position, final):
            caches.append(cache)
            return torch.tensor([0., 0., 2., 3. if position >= 3 else 0.]) if final else None
        runner._text_step = step
        with patch("transformers.DynamicCache", side_effect=lambda **kwargs: object()):
            answer = runner.answer("Q", [], max_tokens=10)
            first = caches[0]
            self.assertEqual(answer["answer_tokens"], 2)
            self.assertFalse(answer["truncated"])
            self.assertTrue(all(cache is first and cache is not original for cache in caches))
            caches.clear()
            limited = runner.answer("Q", [], max_tokens=1)
            self.assertTrue(limited["truncated"])
            self.assertIsNot(caches[0], first)
        self.assertIs(runner.cache, original)

    def test_nonfinite_generation_discards_answer_and_preserves_session(self):
        import torch
        runner = self.language_runner()
        original = runner.cache
        runner._text_step = lambda *args, **kwargs: torch.tensor([0., float("nan"), 1., 0.])
        with patch("transformers.DynamicCache", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                runner.answer("Q", [], max_tokens=3)
        self.assertIs(runner.cache, original)

    def test_prompt_token_limit_is_checked_before_allocating_cache(self):
        runner = self.language_runner()
        runner.tokenizer.apply_chat_template = lambda *args, **kwargs: [1] * 8193
        with patch("transformers.DynamicCache") as cache:
            with self.assertRaisesRegex(ValueError, "8192"):
                runner.answer("Q", [], max_tokens=3)
            cache.assert_not_called()

    def test_state_digest_identifies_tensor_bytes_names_and_layouts(self):
        import torch
        session = CircuitSession()
        state = {"0.recurrent_states.0": torch.tensor([[1., 2.]], dtype=torch.bfloat16),
            "0.conv_states.0": torch.tensor([[0.]])}
        session._state = lambda: state
        digest = session.state_digest()
        state = dict(reversed(list(state.items())))
        self.assertEqual(session.state_digest(), digest)
        original = state["0.recurrent_states.0"]
        for changed in (original + 1, original.reshape(2, 1), original.float()):
            state["0.recurrent_states.0"] = changed
            self.assertNotEqual(session.state_digest(), digest)
        state["0.recurrent_states.0"] = original
        state["1.recurrent_states.0"] = state.pop("0.recurrent_states.0")
        self.assertNotEqual(session.state_digest(), digest)
        state = {}
        with self.assertRaisesRegex(RuntimeError, "Missing navigation state"):
            session.state_digest()

    def mock_merged_runner(self):
        """Mock language/prefix layers; these tests supply no real-model evidence."""
        import torch
        runner = self.language_runner()
        runner.device = "cpu"
        runner.forest = Forest(sample_forest())
        runner.circuit = SimpleNamespace(n_layers=lambda: 5)
        runner.cache = SimpleNamespace(layers=[SimpleNamespace(
            recurrent_states=[torch.tensor([[[[1., 0.]]]])], conv_states=None)])
        runner.meta.update(chip_layers=5, code_dims=list(range(9)))
        runner.navigation_ticks = 17
        runner.weight_id = "mock-weights-not-a-checkpoint"
        return runner

    def test_mock_prefix_tick_counter_increments_only_after_executed_layers(self):
        import torch
        runner = self.mock_merged_runner()
        calls = []
        code = torch.tensor([[[4096. if 267 >> i & 1 else -4096. for i in range(9)]]])
        def layer(h, **kwargs):
            self.assertIs(kwargs["past_key_values"], runner.cache)
            calls.append("prefix")
            return h
        def forbidden(*args, **kwargs):
            self.fail("Navigation entered a language layer")
        runner.model.model = SimpleNamespace(embed_tokens=lambda tokens: code,
            layers=[layer] * 5 + [forbidden])
        with self.assertRaisesRegex(ValueError, "Invalid circuit token"):
            runner.tick(272)
        self.assertEqual(runner.navigation_ticks, 17)
        self.assertEqual(runner.tick(4), 267)
        self.assertEqual(runner.navigation_ticks, 18)
        self.assertEqual(calls, ["prefix"] * 5)

    def test_mock_completion_attests_unchanged_cache_and_rejects_actual_mutation(self):
        import torch
        runner = self.mock_merged_runner()
        state = runner.cache.layers[0].recurrent_states[0]
        mutate = False
        class MockModel:
            config = runner.model.config
            def set_attn_implementation(self, implementation):
                pass
            def __call__(self, tokens, **kwargs):
                self_isolated = kwargs["past_key_values"] is not runner.cache
                self_test.assertTrue(self_isolated)
                if mutate:
                    state.add_(1)
                return SimpleNamespace(logits=torch.tensor([[[0., 0., 1., 2.]]]))
        self_test = self
        runner.model = MockModel()
        before = runner.execution_snapshot()
        with patch("triangle_backend.chat_cache.make_cache", return_value=object()):
            report = runner.complete([{"role": "user", "content": "Q"}], [], max_tokens=3)
            self.assertTrue(report["navigation_unchanged"])
            self.assertEqual(report["forest_sha256"], runner.forest.digest)
            attestation = report["execution"]
            self.assertEqual(attestation["ticks_before"], attestation["ticks_after"])
            self.assertEqual(attestation["state_before"], attestation["state_after"])
            self.assertEqual(runner.execution_snapshot(), before)
            mutate = True
            with self.assertRaisesRegex(RuntimeError, "changed navigation state"):
                runner.complete([{"role": "user", "content": "Q"}], [], max_tokens=3)


if __name__ == "__main__":
    unittest.main()
