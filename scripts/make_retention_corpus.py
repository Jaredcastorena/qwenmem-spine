"""Deterministic synthetic long-session corpus; answer keys never guide retrieval."""
import json
from pathlib import Path
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
tok=AutoTokenizer.from_pretrained(root/'models/qwen3.5-4b',local_files_only=True)
facts=[
 ('deploy_old','Zircon release decision: the active deployment identifier is FALCON-AX-482. This is the current identifier until a later signed correction.'),
 ('recovery','Zircon recovery exception: the offline recovery endpoint must stay on port 7913 even when the main service port changes. Preserve this exception.'),
 ('approval','Zircon open obligation: before deleting the cold archive, obtain SIGNOFF-KELP-29 from Mira. No deletion is authorized without that exact approval.'),
 ('failure','Zircon historical incident: the first migration failed with ERR_KAPPA_62, a checksum mismatch. This is the historical failure code, even after service recovery.'),
 ('preference','Zircon operator preference: exported report dates must use YYYY-DD-MM. This preference applies to reports, not database storage.'),
 ('literal','Zircon exact recovery label is Δ-sable_47/µ. Preserve capitalization, Unicode and punctuation byte for byte.'),
 ('cancel_old','Zircon task: migrate the legacy invoice importer during the next maintenance window.'),
 ('cancel','Zircon correction: the legacy invoice importer migration is CANCELLED. Do not resume that task. This supersedes the earlier migration request.'),
 ('approval_followup','Zircon archive review remains pending. Mira has not provided the required signoff; do not treat routine health checks as authorization.'),
 ('failure_followup','Zircon migration status: service has recovered and current probes pass. This status update does not retract the historical incident report.'),
]
queries=[
 dict(id='current',question='What is the currently active Zircon deployment identifier? Return only the identifier as answer.',support=['deploy_old','deploy_new'],expected='FALCON-AX-482'),
 dict(id='exception',question='What port must the Zircon offline recovery endpoint keep despite a main service change? Return the port number as answer.',support=['recovery'],expected='7913'),
 dict(id='obligation',question='What exact signoff must be obtained before deleting the Zircon cold archive? Return the signoff code as answer.',support=['approval','approval_followup'],expected='SIGNOFF-KELP-29'),
 dict(id='historical',question='What was the error code in the first failed Zircon migration, before service recovered? Return the error code as answer.',support=['failure','failure_followup'],expected='ERR_KAPPA_62'),
 dict(id='preference',question='Which exact date format did the Zircon operator require for exported reports? Return only the format.',support=['preference'],expected='YYYY-DD-MM'),
 dict(id='literal',question='What is the exact Zircon recovery label, preserving Unicode, case and punctuation? Return only the label.',support=['literal'],expected='Δ-sable_47/µ'),
 dict(id='cancelled',question='What is the final status of the Zircon legacy invoice importer migration? Return CANCELLED or ACTIVE.',support=['cancel_old','cancel'],expected='CANCELLED'),
 dict(id='unknown',question='What is the production IPv4 address of the Zircon database? No address may be inferred from other projects.',support=[],expected=None),
]
events=[dict(id=k,text=t,session=0) for k,t in facts]
topics=[('Orchid','garden irrigation'),('Marble','gallery lighting'),('Piper','audio rehearsal'),('Cobalt','warehouse shelving'),('Juniper','trail maintenance'),('Lumen','office scheduling'),('Harbor','library cataloguing'),('Aspen','kitchen inventory')]
templates=[]
for name,topic in topics:
    for variant in range(2):
        text=f'Unrelated project {name}, {topic}: routine review variant {variant}. '
        text+=' '.join([f'The {name} team checked the {topic} worksheet, noted normal observations, and kept its ordinary schedule. The coordinator will review the same checklist at the next regular meeting. These notes concern {name} only.' for _ in range(6)])
        templates.append(text)
count=sum(len(tok.encode(e['text'],add_special_tokens=False)) for e in events)
stages=[]; thresholds=[4096,32768,131072]; corrected=False
while len(stages)<len(thresholds):
    if count>=50000 and not corrected:
        event=dict(id='deploy_new',text='Zircon signed deployment correction: replace FALCON-AX-482 with RAVEN-ZQ-917. The active deployment identifier is now RAVEN-ZQ-917. The older identifier is superseded; the offline recovery exception remains unchanged.',session=len(events)//24)
        corrected=True
    else:
        i=len(events)-len(facts)
        event=dict(id=f'noise-{i:04d}',text=templates[i%len(templates)],session=len(events)//24)
    events.append(event);count+=len(tok.encode(event['text'],add_special_tokens=False))
    if count>=thresholds[len(stages)]:
        stages.append(dict(name=f'{thresholds[len(stages)]//1024}k',history_tokens=count,event_count=len(events),sessions=event['session']+1,current_expected='RAVEN-ZQ-917' if corrected else 'FALCON-AX-482'))
payload=dict(schema=1,seed=0,description='Synthetic facts and 16 repeating unrelated project templates; 24 events per simulated session. Native MiniLM embeddings. No production user data.',events=events,queries=queries,stages=stages)
out=root/'artifacts/retention/corpus.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stages,indent=2))
