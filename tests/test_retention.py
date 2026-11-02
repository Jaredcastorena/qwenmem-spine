import json
from pathlib import Path
import pytest
from triangle_memory.scoring import batch_item,score
from triangle_memory.spine_adapter import resident_forest,replay_query
from triangle_memory.ir_runner import IRRunner

ROOT=Path(__file__).resolve().parents[1]


def test_scoring_accepts_explicit_abstention_and_records_global_citations():
    assert score({'answer':'null','missing':True},None,[])['correct']
    assert not score({'answer':None,'missing':True},'known',[])['correct']
    raw=json.dumps({'answer':{'x':'known','y':None},'evidence_ids':['source'],'missing':True})
    d,scope=batch_item(raw,'x')
    assert scope=='global' and score(d,'known',['source'])['grounded_correct']
    assert not score(d,'known',['another source'])['grounded_correct']
    d,_=batch_item(raw,'y')
    assert score(d,None,[])['correct']


@pytest.mark.parametrize('name',['4k','32k','128k'])
def test_native_spine_export_matches_compiled_ir(name):
    path=ROOT/f'artifacts/retention/native-v2/{name}.json'
    if not path.exists(): pytest.skip('Generate isolated native Spine corpus first')
    data=json.loads(path.read_text());runner=IRRunner(resident_forest(data))
    for query in data['native']['queries']:
        replay_query(runner,query)


@pytest.mark.parametrize('case,needle',[
    ('older_merged_event','AMBER-QN-526'),('long_source_tail','IBIS-TR-804')])
@pytest.mark.xfail(strict=True,reason='Known native renderer loss: last two events / first 1000 characters')
def test_native_renderer_retains_still_binding_source(case,needle):
    path=ROOT/'artifacts/retention/native-boundaries/boundary.json'
    if not path.exists(): pytest.skip('Generate native boundary corpus first')
    data=json.loads(path.read_text())
    assert any(needle in e['text'] for n in data['nodes'].values() for e in n['events'])
    query=next(q for q in data['native']['queries'] if q['id']==case)
    assert any(needle in e['text'] for e in query['cli_evidence'])
