"""Native integration stress cases for old events and long individual source bodies."""
import json
from pathlib import Path
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
tok=AutoTokenizer.from_pretrained(root/'models/qwen3.5-4b',local_files_only=True)
shared=('Copper maintenance exception registry: this dossier tracks the same approval exception. '
        'Routine status notes preserve all earlier exception rules; only an explicit correction can change them. ')*3
events=[
 {'id':'old_exception','session':0,'text':shared+'Binding exception code: AMBER-QN-526. This code remains in force.'},
 {'id':'status_1','session':1,'text':shared+'Routine check completed. No approval exception has been changed.'},
 {'id':'status_2','session':2,'text':shared+'Another routine check completed. All earlier exceptions still apply.'},
 {'id':'long_tail','session':2,'text':'Quartz long maintenance dossier. '+('The long maintenance label is recorded at the end of this original dossier. Routine observations do not replace that label. '*18)+'The exact Quartz long maintenance label is IBIS-TR-804.'},
]
noise=json.loads((root/'artifacts/retention/corpus.json').read_text())['events'][10:26]
events += [dict(id=f'noise-edge-{i}',session=3+i//4,text=e['text']) for i,e in enumerate(noise)]
queries=[
 dict(id='older_merged_event',question='What binding exception code remains in force in the Copper maintenance exception registry? Return only the code.',support=['old_exception'],expected='AMBER-QN-526'),
 dict(id='long_source_tail',question='What exact Quartz long maintenance label is recorded at the end of the original dossier? Return only the label.',support=['long_tail'],expected='IBIS-TR-804'),
]
payload=dict(schema=1,description='Boundary cases: older still-binding event in a merged node, and evidence after the native 1000-character excerpt cutoff.',events=events,queries=queries,
 stages=[dict(name='boundary',history_tokens=sum(len(tok.encode(e['text'],add_special_tokens=False)) for e in events),event_count=len(events),sessions=7)])
(root/'artifacts/retention/boundary-corpus.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload['stages']))
