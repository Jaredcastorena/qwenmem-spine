import copy
import random
from pathlib import Path
import pytest

from triangle_memory.forest import Forest
from triangle_memory.reference import Reference
from triangle_memory.ir_runner import IRRunner
from triangle_memory import protocol as p

FIXTURE = Path(__file__).resolve().parents[1]/"examples/synthetic-forest.json"


@pytest.fixture
def forest():
    return Forest.load(FIXTURE)


def test_nested_alternate_returns_to_exception(forest):
    runner = IRRunner(forest)
    root = forest.slots[forest.roots[0]]
    runner.execute(root)
    runner.execute(p.RIGHT)
    runner.execute(p.RIGHT)
    # The root's alternative survives two descents, and reaches the original exception.
    runner.execute(p.ROUTE)
    result = runner.execute(p.RIGHT)
    apex = next(t-p.APEX for t in result if p.APEX <= t < p.ROUTE)
    assert "recovery endpoint must remain on port 7300" in forest.evidence(apex)["text"]


def test_compiled_vs_independent_reference(forest):
    rng = random.Random(2701)
    runner, oracle = IRRunner(forest), Reference(forest)
    commands = list(range(64))+[p.PEEK,p.LEFT,p.RIGHT]+list(range(p.ROUTE,p.ROUTE+8))
    for _ in range(160):
        cmd = rng.choice(commands)
        assert runner.execute(cmd) == oracle.execute(cmd)


def test_controller_enforces_step_limit(forest):
    runner, oracle = IRRunner(forest), Reference(forest)
    root = forest.slots[forest.roots[0]]
    assert runner.execute(root) == oracle.execute(root)
    # MOVE preserves the budget. Only explicit OPEN starts a new host-budgeted walk.
    for i in range(12):
        assert runner.execute(p.MOVE+root) == oracle.execute(p.MOVE+root)
        got = runner.execute(p.LEFT)
        assert got == oracle.execute(p.LEFT)
        if i >= p.MAX_STEPS:
            assert got[-1] == p.LIMIT


@pytest.mark.parametrize("damage", ["cycle", "dangling", "apex", "coherence"])
def test_reject_invalid_heart_before_installation(forest, damage):
    d = copy.deepcopy(forest.data)
    t = d["triangles"]["triangle:root"]
    if damage == "cycle": t["left"] = "triangle:root"
    if damage == "dangling": t["left"] = "missing"
    if damage == "apex": t["apex"] = "missing"
    if damage == "coherence": d["nodes"]["node:decision"]["vector"] = [-1,0,0]
    with pytest.raises(ValueError): Forest(d)


def test_evidence_budget_is_external_and_exact_bytes(forest):
    with pytest.raises(ValueError, match="budget"):
        forest.evidence(0,max_bytes=1)

