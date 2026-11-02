"""Validated native heart export; stable addresses map to fixed circuit slots."""
from dataclasses import dataclass
import hashlib
import json
import math

from .protocol import CAPACITY


@dataclass(frozen=True)
class Record:
    triangle: bool
    apex: int
    left: int = 0
    right: int = 0


class Forest:
    def __init__(self, data):
        if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
            raise ValueError("Unknown heart export schema")
        self.data = json.loads(json.dumps(data, allow_nan=False))
        self.nodes, self.triangles, self.roots = (self.data.get(k) for k in ("nodes", "triangles", "roots"))
        if not isinstance(self.nodes, dict) or not isinstance(self.triangles, dict) or not isinstance(self.roots, list):
            raise ValueError("Forest requires nodes and triangles objects and a roots array")
        if any(not isinstance(a, str) or not a or len(a.encode()) > 1024 for a in (*self.nodes, *self.triangles)):
            raise ValueError("Invalid forest address")
        if set(self.nodes) & set(self.triangles):
            raise ValueError("Node and triangle addresses overlap")
        self.addresses = sorted(self.nodes) + sorted(self.triangles)
        if len(self.addresses) > CAPACITY:
            raise ValueError(f"Forest must fit {CAPACITY} resident slots")
        self.slots = {address: i for i, address in enumerate(self.addresses)}
        if any(not isinstance(r, str) or r not in self.slots for r in self.roots) or len(set(self.roots)) != len(self.roots):
            raise ValueError("Unresolvable or duplicate root")
        if bool(self.addresses) != bool(self.roots):
            raise ValueError("Nonempty forest requires roots; empty forest requires none")
        for node in self.nodes.values():
            if not isinstance(node, dict) or not isinstance(node.get("text"), str) or not isinstance(node.get("event_id"), str) or not node["event_id"]:
                raise ValueError("Node requires source text and event provenance")
            vector = node.get("vector")
            if (not isinstance(vector, list) or not vector or len(vector) > 8192
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise ValueError("Invalid semantic coordinate")
            norm = sum(x*x for x in vector)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid semantic coordinate norm")
        if len({len(n["vector"]) for n in self.nodes.values()}) > 1:
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
            triangle = self.triangles[address]
            if not isinstance(triangle, dict) or any(not isinstance(triangle.get(k), str) for k in ("apex", "left", "right")):
                raise ValueError("Triangle requires apex, left and right addresses")
            if triangle["apex"] not in self.nodes or triangle["left"] == triangle["right"]:
                raise ValueError("Invalid apex or duplicate branch")
            for side in ("left", "right"):
                branch = triangle[side]
                check(branch)
                coordinate = branch if branch in self.nodes else self.triangles[branch]["apex"]
                if cosine(self.nodes[triangle["apex"]]["vector"], self.nodes[coordinate]["vector"]) < 0.8:
                    raise ValueError("Apex is not coherent with both branches")
            visiting.remove(address)
            visited.add(address)

        for address in self.triangles:
            check(address)
        self.records = {}
        for address, slot in self.slots.items():
            if address in self.nodes:
                self.records[slot] = Record(False, slot)
            else:
                triangle = self.triangles[address]
                self.records[slot] = Record(True, *(self.slots[triangle[k]] for k in ("apex", "left", "right")))
        self.digest = hashlib.sha256(json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cosine(a, b):
    left, right = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
    return sum((x/left)*(y/right) for x, y in zip(a, b))
