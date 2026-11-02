"""Read an isolated native Spine export and replay its query-selected routes in weights.

Spine selects routes with its actual MiniLM embeddings. This adapter does not use
benchmark answers to select a route, and keeps all node event provenance available.
"""
from . import protocol as p
from .forest import Forest


def resident_forest(export):
    return Forest({k:export[k] for k in ('schema','nodes','triangles','roots')})


def replay_query(runner,query):
    forest=runner.forest
    visited=[]
    traces=[]
    for walk in query['walks']:
        coordinates=[];alternates=[]
        commands=[forest.slots[walk['root']],*[p.command(c) for c in walk['commands']]]
        for cmd in commands:
            trace=runner.execute(cmd);traces.append([p.name(t) for t in trace])
            for token in trace:
                if p.APEX<=token<p.ROUTE:
                    address=forest.addresses[token-p.APEX]
                    if address not in coordinates: coordinates.append(address)
                if p.ALT<=token<p.APEX:
                    alternates.append(forest.addresses[token-p.ALT])
        if coordinates!=walk['coordinates']:
            raise AssertionError({'model':coordinates,'native':walk['coordinates'],'traces':traces})
        # Native core may retain one additional route at depth 8 before truncating.
        if alternates!=walk['alternates'][:len(alternates)]:
            raise AssertionError('Native/model alternate route mismatch')
        for address in coordinates:
            if address not in visited: visited.append(address)
    evidence=[]
    for address in visited:
        for event in reversed(forest.nodes[address]['events'][-2:]):
            text=event['text']
            if len(text)>1000: text=text[:1000]+'\n[truncated]'
            evidence.append(dict(node_id=address.removeprefix('node:'),event_id=event['event_id'],text=text))
            if len(evidence)>=12: break
        if len(evidence)>=12: break
    native=[{k:e[k] for k in ('node_id','event_id','text')} for e in query['cli_evidence']]
    if evidence!=native:
        raise AssertionError('Compiled retrieval differs from the actual Spine CLI renderer')
    return evidence,traces
