"""Bounded command execution and validated, private, atomic circuit snapshots."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import protocol as p

PROTOCOL_ID = hashlib.sha256(json.dumps({"schema": 1, "capacity": p.CAPACITY,
    "max_depth": p.MAX_STEPS, "vocab_size": p.VOCAB_SIZE,
    "tokens": [p.OPEN, p.MOVE, p.ALT, p.APEX, p.ROUTE, p.PEEK, p.LEFT, p.RIGHT,
               p.READY, p.INVALID, p.LIMIT, p.LEAF, p.PAD]}, sort_keys=True).encode()).hexdigest()


def tensor_bytes(tensor):
    import torch
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class CircuitSession:
    def execute(self, token):
        trace = [token]
        for _ in range(8):
            token = self.tick(token)
            trace.append(token)
            if token in p.STOP:
                return trace
        raise RuntimeError("Compiled trace exceeded independent host limit")

    def _state(self):
        result = {}
        for i, layer in enumerate(self.cache.layers[:self.circuit.n_layers()]):
            for name in ("recurrent_states", "conv_states"):
                value = getattr(layer, name)
                if isinstance(value, (list, tuple, dict)):
                    items = value.items() if isinstance(value, dict) else enumerate(value)
                    for j, tensor in items:
                        if tensor is not None:
                            result[f"{i}.{name}.{j}"] = tensor.detach().cpu().contiguous().clone()
                elif value is not None:
                    result[f"{i}.{name}"] = value.detach().cpu().contiguous().clone()
        return result

    def state_digest(self):
        """Identify actual prefix cache tensors, including their names and layouts."""
        state = self._state()
        if not state:
            raise RuntimeError("Missing navigation state")
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            layout = json.dumps([key, str(value.dtype), list(value.shape)],
                separators=(",", ":")).encode()
            payload = tensor_bytes(value)
            for part in (layout, payload):
                digest.update(len(part).to_bytes(8, "big"))
                digest.update(part)
        return digest.hexdigest()

    def save(self, path):
        import torch
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tensors = self._state()
        if not tensors or any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError("Cannot save missing or non-finite circuit state")
        metadata = {"schema": "1", "protocol_sha256": PROTOCOL_ID,
            "forest_sha256": self.forest.digest, "weights_sha256": self.weight_id,
            "checksums": json.dumps({k: hashlib.sha256(tensor_bytes(v)).hexdigest()
                                      for k, v in tensors.items()}, sort_keys=True)}
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".triangle-state-")
        os.close(fd)
        try:
            save_file(tensors, temporary, metadata=metadata)
            os.chmod(temporary, 0o600)
            with open(temporary, "rb") as saved:
                os.fsync(saved.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def restore(self, path):
        import torch
        from safetensors import safe_open
        expected = self._state()
        with safe_open(path, framework="pt", device="cpu") as saved:
            meta = saved.metadata() or {}
            if meta.get("schema") != "1" or meta.get("protocol_sha256") != PROTOCOL_ID:
                raise ValueError("Snapshot protocol or schema differs")
            if meta.get("forest_sha256") != self.forest.digest:
                raise ValueError("Snapshot belongs to another forest")
            if meta.get("weights_sha256") != self.weight_id:
                raise ValueError("Snapshot belongs to different compiled weights")
            if set(saved.keys()) != set(expected):
                raise ValueError("Snapshot state keys differ")
            checksums = json.loads(meta.get("checksums", "{}"))
            if not isinstance(checksums, dict) or set(checksums) != set(expected):
                raise ValueError("Snapshot checksum keys differ")
            tensors = {key: saved.get_tensor(key) for key in saved.keys()}
        for key, value in tensors.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError("Snapshot tensor layout differs")
            if not bool(torch.isfinite(value).all()) or hashlib.sha256(tensor_bytes(value)).hexdigest() != checksums[key]:
                raise ValueError("Snapshot tensor is damaged")
        # Allocate every device tensor before touching live recurrent state.
        prepared = {key: value.to(self.device) for key, value in tensors.items()}
        for key, value in prepared.items():
            layer, name, *index = key.split(".")
            target = getattr(self.cache.layers[int(layer)], name)
            (target[int(index[0])] if index else target).copy_(value)
