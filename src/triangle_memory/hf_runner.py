"""The triangle circuit as real Qwen3.5 weights and persistent recurrent state."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import circuit, protocol as p


class ModelRunner:
    def __init__(self, forest, *, device="cpu", model_path=None):
        import torch
        from chipformer.q35 import tiny
        from chipformer.rv.hf import inject
        from transformers import Qwen3_5ForCausalLM

        self.forest = forest
        self.circuit = circuit.build()
        circuit.install(self.circuit, forest)
        self.device = device
        torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
        if model_path:
            self.model = Qwen3_5ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                         attn_implementation="eager")
        else:
            self.model = tiny.build(self.circuit, list(range(p.VOCAB_SIZE)),
                [next(iter(e.terms)) for e in self.circuit.code], dtype=torch.bfloat16)
        self.model.config._attn_implementation = "eager"
        self.model = self.model.to(device).eval()
        self.weight_id = self._weight_digest()
        # The padding embedding above VOCAB_SIZE is identically zero.
        with torch.no_grad():
            self.cache = self.model(torch.tensor([[p.VOCAB_SIZE]], device=device),
                                    use_cache=True).past_key_values
        inject(self.circuit, self.cache)

    def _weight_digest(self):
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor_bytes(tensor))
        return h.hexdigest()

    def tick(self, token):
        import torch
        if not 0 <= token < p.VOCAB_SIZE:
            raise ValueError("Invalid circuit token")
        with torch.no_grad():
            result = self.model(torch.tensor([[token]], device=self.device),
                                past_key_values=self.cache, use_cache=True)
            self.cache = result.past_key_values
            logits = result.logits[0,-1,:p.VOCAB_SIZE]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Non-finite circuit logits")
            return int(logits.argmax())

    def execute(self, token):
        trace = [token]
        for _ in range(8):
            token = self.tick(token)
            trace.append(token)
            if token in p.STOP:
                return trace
        raise RuntimeError("Compiled trace exceeded independent host limit")

    def export(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        (path/"triangle.json").write_text(json.dumps({"schema":1,"capacity":p.CAPACITY,
            "max_steps":p.MAX_STEPS,"weights_sha256":self.weight_id,
            "circuit":self.circuit.stats()}, indent=2)+"\n")

    def _state(self):
        result = {}
        for i, layer in enumerate(self.cache.layers[:self.circuit.n_layers()]):
            for name in ("recurrent_states", "conv_states"):
                value = getattr(layer, name)
                if isinstance(value, (list,tuple,dict)):
                    items = value.items() if isinstance(value,dict) else enumerate(value)
                    for j, tensor in items:
                        if tensor is not None:
                            result[f"{i}.{name}.{j}"] = tensor.detach().cpu().contiguous()
                elif value is not None:
                    result[f"{i}.{name}"] = value.detach().cpu().contiguous()
        return result

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = self._state()
        metadata = {"schema":"1", "forest_sha256":self.forest.digest,
                    "weights_sha256":self.weight_id,
                    "checksums":json.dumps({k:hashlib.sha256(tensor_bytes(v)).hexdigest()
                                             for k,v in tensors.items()},sort_keys=True)}
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".triangle-state-")
        os.close(fd)
        try:
            save_file(tensors, temp, metadata=metadata)
            os.chmod(temp, 0o600)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def restore(self, path):
        import torch
        from safetensors import safe_open
        expected = self._state()
        # Validate the entire snapshot before mutating live state.
        with safe_open(path, framework="pt", device="cpu") as saved:
            meta = saved.metadata()
            if meta.get("schema") != "1" or meta.get("forest_sha256") != self.forest.digest:
                raise ValueError("Snapshot belongs to another forest or schema")
            if meta.get("weights_sha256") != self.weight_id:
                raise ValueError("Snapshot belongs to different compiled weights")
            if set(saved.keys()) != set(expected):
                raise ValueError("Snapshot state keys differ")
            checksums = json.loads(meta["checksums"])
            tensors = {k:saved.get_tensor(k) for k in saved.keys()}
        for key, value in tensors.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError("Snapshot tensor layout differs")
            if not torch.isfinite(value).all() or hashlib.sha256(tensor_bytes(value)).hexdigest() != checksums.get(key):
                raise ValueError("Snapshot tensor is damaged")
        for key,value in tensors.items():
            layer, name, *index = key.split(".")
            target = getattr(self.cache.layers[int(layer)], name)
            if index:
                target[int(index[0])].copy_(value.to(self.device))
            else:
                target.copy_(value.to(self.device))


def tensor_bytes(tensor):
    import torch
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

