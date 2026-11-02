import os
from pathlib import Path
import pytest

from triangle_memory.forest import Forest
from triangle_memory.hf_runner import ModelRunner
from triangle_memory.reference import Reference
from triangle_memory import protocol as p


@pytest.fixture(scope="module")
def forest():
    return Forest.load(Path(__file__).resolve().parents[1]/"examples/synthetic-forest.json")


def test_real_qwen_weights_match_reference_and_reopen(forest,tmp_path):
    device = os.environ.get("TRIANGLE_TEST_DEVICE", "cpu")
    runner, oracle = ModelRunner(forest,device=device), Reference(forest)
    root = forest.slots[forest.roots[0]]
    for cmd in [root,p.RIGHT,p.RIGHT,p.PEEK]:
        assert runner.execute(cmd) == oracle.execute(cmd)
    runner.save(tmp_path/"session.safetensors")
    runner.export(tmp_path/"model")
    reopened = ModelRunner(forest,device=device,model_path=tmp_path/"model")
    reopened.restore(tmp_path/"session.safetensors")
    for cmd in [p.ROUTE,p.RIGHT,p.PEEK,63,p.PEEK,root,p.ROUTE+7]:
        assert reopened.execute(cmd) == oracle.execute(cmd)


def test_snapshot_cannot_be_rebound_to_changed_heart(forest,tmp_path):
    runner=ModelRunner(forest)
    runner.save(tmp_path/"session.safetensors")
    import copy
    d=copy.deepcopy(forest.data)
    d["nodes"]["node:decision"]["text"]="changed"
    other=ModelRunner(Forest(d))
    with pytest.raises(ValueError,match="forest"):
        other.restore(tmp_path/"session.safetensors")


def test_random_bf16_sequences_and_snapshot_corruption(forest,tmp_path):
    import random
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    runner=ModelRunner(forest,device=os.environ.get("TRIANGLE_TEST_DEVICE","cpu"))
    oracle=Reference(forest)
    rng=random.Random(173)
    root=forest.slots[forest.roots[0]]
    commands=[root,p.LEFT,p.RIGHT,p.PEEK,*range(p.ROUTE,p.ROUTE+p.MAX_STEPS),63]
    for cmd in [root]+[rng.choice(commands) for _ in range(128)]:
        assert runner.execute(cmd)==oracle.execute(cmd)
    path=tmp_path/"state.safetensors"
    runner.save(path)
    with safe_open(path,framework="pt") as f:
        metadata=f.metadata()
        data={k:f.get_tensor(k) for k in f.keys()}
    key=next(k for k in data if "recurrent_states" in k)
    # Ballast can be ~1e38: adding one would round away in float32.
    flat=data[key].view(-1)
    flat[0]=-flat[0] if flat[0] != 0 else 1
    save_file(data,path,metadata=metadata)
    before=runner._state()
    with pytest.raises(ValueError,match="damaged"):
        runner.restore(path)
    assert all(torch.equal(v,runner._state()[k]) for k,v in before.items())


def test_full_resident_bank_and_eight_step_boundary():
    nodes={f"n:{i}":{"text":f"source {i}","event_id":f"e:{i}","vector":[1,0]} for i in range(2)}
    triangles={}
    previous="n:1"
    for i in range(62):
        name=f"t:{i:02d}"
        triangles[name]={"apex":"n:1","left":previous,"right":"n:0"}
        previous=name
    forest=Forest({"schema":1,"nodes":nodes,"triangles":triangles,"roots":[previous]})
    assert forest.slots[previous]==63
    runner=ModelRunner(forest,device=os.environ.get("TRIANGLE_TEST_DEVICE","cpu"))
    oracle=Reference(forest)
    assert runner.execute(63)==oracle.execute(63)
    for i in range(10):
        trace=runner.execute(p.LEFT)
        assert trace==oracle.execute(p.LEFT)
        assert (trace[-1]==p.LIMIT)==(i>=8)
    for cmd in (p.ROUTE,p.LEFT,63,p.RIGHT,p.LEFT):
        assert runner.execute(cmd)==oracle.execute(cmd)
