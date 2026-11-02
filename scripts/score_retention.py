"""Derive content/grounding scores without altering raw model outputs or strict scores."""
import argparse,json
from pathlib import Path
from triangle_memory.scoring import batch_item,score

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--native',default='artifacts/retention/native-v2');args=p.parse_args()
folder=Path(args.directory);rows=[]
for data in json.loads((folder/'results.json').read_text()):
    native=json.loads((Path(args.native)/(data['stage']['name']+'.json')).read_text())
    originals={e['event_id']:e['text'] for n in native['nodes'].values() for e in n['events']}
    stage={'stage':data['stage'],'resident_records':data['resident_records'],'installed':data['installed'],'conditions':{},'trace_parity':[],'cases':[]}
    for case in data['cases']:
        answer_sources=[e for e in case['support_event_ids'] if case['expected'] is not None and case['expected'] in originals.get(e,'')]
        row={'id':case['id'],'conditions':{},'support_coverage':case['support_coverage']}
        stage['trace_parity'].append(case.get('trace_parity',False))
        for condition,result in case['conditions'].items():
            graded=score(result['decoded'],case['expected'],answer_sources)
            graded.update(prompt_tokens=result['prompt_tokens'],answer_tokens=result['answer_tokens'],seconds=result['seconds'],citation_scope='per_question',strict_correct=result['correct'])
            row['conditions'][condition]=graded
        if 'full_history' in data:
            decoded,scope=batch_item(data['full_history']['answer'],case['id'])
            graded=score(decoded,case['expected'],answer_sources);graded['citation_scope']=scope
            row['conditions']['full_history']=graded
        stage['cases'].append(row)
    for condition in stage['cases'][0]['conditions']:
        values=[c['conditions'][condition] for c in stage['cases']]
        stage['conditions'][condition]={'correct':sum(v['correct'] for v in values),'grounded_correct':sum(v['grounded_correct'] for v in values),'total':len(values)}
        if condition!='full_history':
            stage['conditions'][condition]['prompt_tokens_range']=[min(v['prompt_tokens'] for v in values),max(v['prompt_tokens'] for v in values)]
    if 'full_history' in data:
        stage['full_history']={k:data['full_history'][k] for k in ('prompt_tokens','answer_tokens','seconds','triangle_state_unchanged','truncated')}
    rows.append(stage)
(folder/'scored.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for s in rows:
    print(s['stage']['name'],s['conditions'],s.get('full_history'))
