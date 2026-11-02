import json,os
from pathlib import Path
import pytest
from triangle_memory import protocol as p
from triangle_memory.forest import Forest
from triangle_memory.hf_runner import ModelRunner


def test_deep_native_compaction_in_real_bf16_layers():
    file=Path(__file__).resolve().parents[1]/'artifacts/retention/native-geometry.json'
    if not file.exists(): pytest.skip('Build native geometry fixture first')
    data=json.loads(file.read_text())
    forest=Forest({k:data[k] for k in ('schema','nodes','triangles','roots')})
    assert len(forest.records)==63 and len(forest.triangles)==31
    runner=ModelRunner(forest,device=os.environ.get('TRIANGLE_TEST_DEVICE','cpu'))
    for query in data['geometry']['queries']:
        coordinates=[];alternates=[]
        for cmd in [forest.slots[forest.roots[0]],*[p.command(c) for c in query['commands']]]:
            for token in runner.execute(cmd):
                if p.APEX<=token<p.ROUTE:
                    a=forest.addresses[token-p.APEX]
                    if a not in coordinates: coordinates.append(a)
                if p.ALT<=token<p.APEX: alternates.append(forest.addresses[token-p.ALT])
        assert coordinates==query['coordinates']
        assert alternates==query['alternates']
