import json
from triangle_memory import raw_audit


def test_source_audit_rejects_wrong_and_invented_citations_and_changed_unicode(monkeypatch):
    events=[dict(id='gold',text='Literal Δ-sable_47/µ'),dict(id='noise',text='Unrelated source')]
    query=dict(id='literal',expected='Δ-sable_47/µ',support=['gold'])
    monkeypatch.setattr(raw_audit,'control_inputs',lambda stage: ({},events,{'gold':'event:1','noise':'event:2'},[query],[]))
    def verify(answer,ids):
        text=json.dumps({'literal':dict(answer=answer,evidence_ids=ids,missing=False)})
        return raw_audit.audit_answer(text,'fixture')['verified']
    assert verify('Δ-sable_47/µ',['event:1'])==1
    assert verify('Δ-sable_47/µ',['event:2'])==0
    assert verify('Δ-sable_47/µ',['event:1','invented'])==0
    assert verify('Δ-sable_47/u',['event:1'])==0
