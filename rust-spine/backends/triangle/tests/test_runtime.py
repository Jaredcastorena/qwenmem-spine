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
        self.assertEqual(runner.execute(4), [4, 192, 267])
        self.assertEqual(runner.execute(265), [265, 130, 67, 192, 267])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.safetensors"
            runner.save(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            runner.reset()
            runner.restore(path)
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


if __name__ == "__main__":
    unittest.main()
