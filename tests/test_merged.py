"""Explicit opt-in: requires the locally merged 4B checkpoint and about 12GB device RAM."""
import os
from pathlib import Path
import pytest

from triangle_memory import protocol as p
from triangle_memory.forest import Forest
from triangle_memory.reference import Reference


@pytest.mark.skipif(not os.environ.get("TRIANGLE_MERGED_TEST"),reason="opt-in 4B checkpoint test")
def test_merged_prefix_and_fresh_language_session(tmp_path):
    from triangle_memory.merged_runner import MergedRunner
    forest=Forest.load("examples/synthetic-forest.json")
    runner=MergedRunner(forest,model_path="models/qwen3.5-triangles",device=os.environ.get("TRIANGLE_TEST_DEVICE","cpu"))
    oracle=Reference(forest)
    root=forest.slots[forest.roots[0]]
    for cmd in (root,p.RIGHT,p.RIGHT):
        assert runner.execute(cmd)==oracle.execute(cmd)
    runner.save(tmp_path/"session.safetensors")
    runner.reset_session()
    runner.restore(tmp_path/"session.safetensors")
    for cmd in (p.ROUTE,p.RIGHT):
        assert runner.execute(cmd)==oracle.execute(cmd)
    evidence=[forest.evidence(forest.slots["node:exception"])]
    before=runner._state()
    result=runner.answer("What port must the offline recovery endpoint use? Cite the event.",evidence,max_tokens=100)
    assert "7300" in result["answer"] and "event:exception" in result["answer"],result
    after=runner._state()
    assert all((before[k]==after[k]).all() for k in before)
    Path("artifacts/merged-language-answer.json").write_text(__import__("json").dumps(result,indent=2)+"\n")
