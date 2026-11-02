"""Content scoring independent of JSON layout; raw strict scores remain in run reports."""
import json


def decode_object(text):
    try:
        return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]
    except (ValueError,TypeError):
        return {}


def batch_item(text,key):
    data=decode_object(text)
    if isinstance(data.get(key),dict):
        return data[key], 'per_question'
    # The model can put the answer map under one outer answer field and list
    # all citations globally. Preserve that distinction instead of marking
    # every correctly recalled value wrong because of envelope formatting.
    if isinstance(data.get('answer'),dict) and key in data['answer']:
        value=data['answer'][key]
        return dict(answer=value,missing=value is None,evidence_ids=data.get('evidence_ids',[])), 'global'
    return {}, 'invalid'


def score(decoded,expected,support_ids):
    answer=decoded.get('answer')
    absent=answer is None or isinstance(answer,str) and answer.strip().lower()=='null'
    abstained=decoded.get('missing') is True and absent
    correct=abstained if expected is None else (decoded.get('missing') is not True and str(answer)==expected)
    cited=decoded.get('evidence_ids',[])
    support=bool(set(cited)&set(support_ids)) if isinstance(cited,list) else False
    return dict(correct=correct,abstained=abstained,support_cited=support,
                grounded_correct=correct and (expected is None or support))
