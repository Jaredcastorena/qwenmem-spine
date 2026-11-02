"""Recheck raw answers and source citations independently of benchmark grades."""
import argparse,json
from pathlib import Path
from triangle_memory.raw_audit import audit_answer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',default='artifacts/retention/full-sdpa')
args=p.parse_args();directory=Path(args.directory)
rows=[]
for stage in ('4k','32k','128k'):
    file=directory/f'{stage}.json'
    if not file.exists():
        rows.append(dict(stage=stage,status='no completed answer; cannot verify recall'))
        continue
    data=json.loads(file.read_text())
    result=audit_answer(data['answer'],stage)
    result.update(source=str(file),prompt_tokens=data['prompt_tokens'],truncated=data['truncated'])
    rows.append(result)
(directory/'independent-audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for row in rows: print(row.get('stage'),row.get('verified'),row.get('total'),row.get('status','audited'))
