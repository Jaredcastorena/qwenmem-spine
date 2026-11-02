"""Verified heart export. Stable addresses stay outside the small hardware slot space."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .protocol import CAPACITY


@dataclass(frozen=True)
class Record:
    triangle: bool
    apex: int
    left: int = 0
    right: int = 0


class Forest:
    def __init__(self, data):
        # Copy at the trust boundary; callers cannot mutate a verified installation afterward.
        self.data = json.loads(json.dumps(data, allow_nan=False))
        if self.data.get("schema") != 1:
            raise ValueError("Unknown heart export schema")
        self.nodes = self.data["nodes"]
        self.triangles = self.data["triangles"]
        if set(self.nodes) & set(self.triangles):
            raise ValueError("Node and triangle addresses overlap")
        self.addresses = sorted(self.nodes) + sorted(self.triangles)
        if not self.addresses or len(self.addresses) > CAPACITY:
            raise ValueError(f"Forest must fit {CAPACITY} resident slots")
        self.slots = {address: i for i, address in enumerate(self.addresses)}
        self.roots = self.data["roots"]
        if not self.roots or any(r not in self.slots for r in self.roots):
            raise ValueError("Unresolvable root")
        for n in self.nodes.values():
            if not isinstance(n.get("text"), str) or not n.get("event_id"):
                raise ValueError("Node requires source text and event provenance")
            v = n["vector"]
            if not v or any(not math.isfinite(x) for x in v) or sum(x*x for x in v) == 0:
                raise ValueError("Invalid semantic coordinate")
        if len({len(n["vector"]) for n in self.nodes.values()}) != 1:
            raise ValueError("Embedding dimensions differ")
        visiting, visited = set(), set()

        def check(address):
            if address in visiting:
                raise ValueError("Triangle cycle")
            if address in visited or address in self.nodes:
                return
            if address not in self.triangles:
                raise ValueError("Dangling branch")
            visiting.add(address)
            t = self.triangles[address]
            if t["apex"] not in self.nodes or t["left"] == t["right"]:
                raise ValueError("Invalid apex or duplicate branch")
            for side in ("left", "right"):
                check(t[side])
                branch = t[side]
                coord = branch if branch in self.nodes else self.triangles[branch]["apex"]
                if cosine(self.nodes[t["apex"]]["vector"], self.nodes[coord]["vector"]) < 0.8:
                    raise ValueError("Apex is not coherent with both branches")
            visiting.remove(address)
            visited.add(address)

        for t in self.triangles:
            check(t)
        self.records = {}
        for a, slot in self.slots.items():
            if a in self.nodes:
                self.records[slot] = Record(False, slot)
            else:
                t = self.triangles[a]
                self.records[slot] = Record(True, self.slots[t["apex"]], self.slots[t["left"]], self.slots[t["right"]])
        self.digest = hashlib.sha256(json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def evidence(self, slot, *, max_bytes=4096):
        if slot not in self.records or self.records[slot].triangle:
            raise ValueError("Evidence address is not a node")
        address = self.addresses[slot]
        node = self.nodes[address]
        if len(node["text"].encode()) > max_bytes:
            raise ValueError("Evidence exceeds host byte budget; request a smaller source segment")
        return {"node_id": address, "event_id": node["event_id"], "text": node["text"]}


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))

