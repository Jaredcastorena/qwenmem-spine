#!/usr/bin/env python3
"""Prove internal state residency and identify the circuit in the actual checkpoint.

CPU only. Regenerates only the five compiled prefix layers, never the language
model. Use --state-only for the small CPU circuit; default also reads the merged checkpoint.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rust-spine/backends/triangle"))
from triangle_backend.upstream import configure, enable
configure(ROOT / "vendor/qwenputer")
enable()


def weight_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return result.hexdigest()


def verify_checkpoint():
    import torch
    from safetensors import safe_open
    from chipformer.q35.ir import Target
    from chipformer.q35.emit import Layout, emit_layers, embedding_rows
    from chipformer.q35.merge import pick_hidden
    from triangle_backend import circuit, protocol as p
    model = ROOT / "models/qwen3.5-triangles"
    cfg = json.loads((model / "config.json").read_text())
    meta = json.loads((model / "triangle.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    compiled = circuit.build(Target(cfg["linear_num_key_heads"],
        cfg["linear_num_value_heads"], cfg["intermediate_size"]))
    hidden, ballast = pick_hidden(meta["original_hidden"], len(compiled.dims))
    assert hidden == cfg["hidden_size"]
    assert compiled.n_layers() == meta["chip_layers"] == 5
    layout = Layout(hidden=hidden, off=meta["original_hidden"], m=ballast)
    code_dims = [layout.dim(next(iter(expr.terms))) for expr in compiled.code]
    assert code_dims == meta["code_dims"]
    emitted = emit_layers(compiled, layout, cfg, 0, "model.", keep=compiled.keep)
    expected_keys = {name for name in index if name.startswith("model.layers.")
                     and int(name.split(".")[2]) < meta["chip_layers"]}
    assert set(emitted) == expected_keys
    count, total_bytes = 0, 0
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(model / name, framework="pt"))
                   for name in {index[key] for key in expected_keys} | {index["model.embed_tokens.weight"]}}
        for name in sorted(expected_keys):
            expected = emitted.pop(name).to(torch.bfloat16)
            actual = readers[index[name]].get_tensor(name)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
            assert torch.equal(actual, expected), name
            count += 1
            total_bytes += actual.numel() * actual.element_size()
        expected = embedding_rows(compiled, layout, list(range(p.VOCAB_SIZE)), hidden).to(torch.bfloat16)
        actual = readers[index["model.embed_tokens.weight"]].get_slice("model.embed_tokens.weight")[meta["text_vocab"]:, :]
        assert torch.equal(actual, expected)
    return {"compiled_layers": 5, "prefix_tensors_equal": count,
            "prefix_tensor_bytes": total_bytes, "command_embedding_rows_equal": p.VOCAB_SIZE,
            "code_dims_equal": code_dims, "weights_sha256": meta["weights_sha256"],
            "language_model_loaded": False, "device": "cpu"}


def verify_state():
    import torch
    from chipformer.q35 import tiny
    from chipformer.rv.hf import inject, head_slots
    from triangle_backend import circuit, protocol as p
    from triangle_backend.forest import Forest
    from triangle_backend.session import CircuitSession
    data = {"schema": 1, "nodes": {name: {"text": f"Source {name}",
        "event_id": f"event:{name}", "vector": [1.0, 0.0]} for name in ("a", "b", "c")},
        "triangles": {"d": {"apex": "a", "left": "a", "right": "b"},
                      "e": {"apex": "a", "left": "d", "right": "c"}}, "roots": ["e"]}

    class Session(CircuitSession):
        def __init__(self):
            self.device = "cpu"
            self.forest = Forest(data)
            self.circuit = circuit.build()
            circuit.install(self.circuit, self.forest)
            self.model = tiny.build(self.circuit, list(range(p.VOCAB_SIZE)),
                [next(iter(e.terms)) for e in self.circuit.code], dtype=torch.bfloat16)
            self.model.config._attn_implementation = "eager"
            self.model.eval()
            self.weight_id = weight_digest(self.model)
            with torch.no_grad():
                self.cache = self.model(torch.tensor([[p.VOCAB_SIZE]]), use_cache=True).past_key_values
                inject(self.circuit, self.cache)

        def tick(self, token):
            with torch.no_grad():
                result = self.model(torch.tensor([[token]]), past_key_values=self.cache, use_cache=True)
                self.cache = result.past_key_values
                logits = result.logits[0, -1, :p.VOCAB_SIZE]
                assert torch.isfinite(logits).all()
                return int(logits.argmax())

    class NoHostForest:
        def __init__(self, digest):
            self.digest = digest

        def __getattr__(self, name):
            raise AssertionError(f"Navigation tried to read host forest: {name}")

    runner = Session()
    weights_before = runner.weight_id
    records_head = runner.circuit.records_head
    node_index = next(i for i, node in enumerate(runner.circuit.nodes) if node is records_head)
    layer, head = head_slots(runner.circuit)[node_index]
    ratio = runner.circuit.target.value_heads // runner.circuit.target.key_heads

    def resident_records():
        return runner.cache.layers[layer].recurrent_states[0][
            0, head * ratio:head * ratio + len(records_head.vheads())]

    resident_before = resident_records().clone()
    runner.forest = NoHostForest(runner.forest.digest)
    for node in runner.circuit.nodes:
        if hasattr(node, "init"):
            node.init = []
    traces = []
    for command, expected in ((4, [4, 192, 267]), (265, [265, 130, 67, 192, 267])):
        actual = runner.execute(command)
        assert actual == expected, (actual, expected)
        traces.append(actual)
    with tempfile.TemporaryDirectory(prefix="triangle-residency-audit-") as directory:
        path = Path(directory) / "navigation.safetensors"
        runner.save(path)
        snapshot_bytes = path.stat().st_size
        for command, expected in ((266, [266, 128, 65, 193, 267]), (256, [256, 66, 194, 267])):
            actual = runner.execute(command)
            assert actual == expected, (actual, expected)
            traces.append(actual)
        assert torch.equal(resident_before, resident_records())
        resident_records().zero_()
        assert runner.execute(4) == [4, p.INVALID]
        runner.restore(path)
        assert runner.execute(266) == [266, 128, 65, 193, 267]
        assert torch.equal(resident_before, resident_records())
    assert weight_digest(runner.model) == weights_before
    # The installed registry has remained read-only while cursor/routes changed.
    return {"real_bf16_compiled_circuit": True, "host_forest_lookup_blocked": True,
            "all_host_initializers_removed_after_install": True, "exact_traces": traces,
            "resident_records_unchanged_by_navigation": True,
            "zeroing_resident_records_makes_open_invalid": True,
            "snapshot_restores_records_and_cursor_without_host_forest": True,
            "weights_unchanged": True, "snapshot_bytes": snapshot_bytes,
            "fixture": "small synthesized Qwen circuit; not a new full-checkpoint generation test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-only", action="store_true", help="Skip the external 4B checkpoint comparison")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    checkpoint = None if args.state_only else verify_checkpoint()
    if checkpoint is not None:
        print("Actual merged checkpoint prefix matches the regenerated navigation circuit", flush=True)
    state = verify_state()
    result = {"schema": 1, "checkpoint": checkpoint, "state_residency": state,
              "full_native_triangle_lifecycle_in_model": False,
              "host_operations": ["create", "coherence", "compact", "root_ranking",
                                  "semantic_branch_selection", "source_rehydration", "install"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Recurrent-state residency, read-only registry and snapshot recovery verified", flush=True)


if __name__ == "__main__":
    main()
