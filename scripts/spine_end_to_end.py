"""Native browser/agent acceptance; run only through scripts/guarded-python.sh.

Uses a disposable copy of the synthetic 128K heart, never a personal heart.
The public chat API, model tools, journal, grounding and projection remain enabled.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
from urllib.request import Request, urlopen
ROOT=Path(__file__).resolve().parents[1]
BINARY=Path(os.environ.get('SPINE_TEST_BINARY',str(ROOT/'artifacts/spine-target/debug/spine')))
FORK=ROOT/'rust-spine'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_navigation_events(paths):
    events=[]
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            # The terminal prompt has no newline; stderr audit JSON may follow it.
            line=line.removeprefix('you> ')
            try: event=json.loads(line)
            except (ValueError,TypeError): continue
            if isinstance(event,dict): events.append(event)
    return events

def navigation_evidence(events, checkpoint, completed, require_branch):
    chats=[e for e in events if e.get('type')=='triangle_chat']
    banks=[e for e in events if e.get('type')=='triangle_internal_bank']
    bound=True; pending=[]
    for event in events:
        if event.get('type')=='triangle_internal_bank': pending.append(event)
        if event.get('type')!='triangle_chat': continue
        count=event.get('bank_count',0)
        selected=pending[-count:] if count else []
        # A cancelled request may leave a complete bank group with no chat event.
        # Bind each accepted completion to its immediately preceding bank group.
        bound &= len(selected)==count and [b.get('bank') for b in selected]==list(range(count))
        bound &= sum(b.get('resident_records',0) for b in selected)==event.get('resident_records')
        bound &= sum(b.get('resident_triangles',0) for b in selected)==event.get('resident_triangles')
        bound &= sum(c in ('left','right') for b in selected for c in b.get('commands',[]))==event.get('branch_commands')
        if selected:
            final=selected[-1]; execution=final.get('execution',[])
            bound &= bool(execution) and final.get('forest_sha256')==event.get('forest_sha256')
            if execution:
                bound &= execution[-1].get('state_after')==event.get('execution',{}).get('state_before')
                bound &= execution[-1].get('ticks_after')==event.get('execution',{}).get('ticks_before')
        pending=[]
    proof={
        'completed_model_rounds':len(chats),
        'bank_executions':sum(c.get('bank_count',0) for c in chats),
        'resident_triangle_installations':sum(c.get('resident_triangles',0) for c in chats),
        'branch_commands':sum(c.get('branch_commands',0) for c in chats),
        'recorded_bank_executions':len(banks),
        'recorded_triangle_installations':sum(b.get('resident_triangles',0) for b in banks),
        'recorded_branch_commands':sum(c in ('left','right') for b in banks for c in b.get('commands',[])),
        'completed_rounds_bound_to_banks':bool(chats) and bound,
        'all_completions_gated':bool(chats) and all(c.get('retrieval')=='compiled_native_handoff' and c.get('source_gate_verified') is True and c.get('resident_records',0)>0 and c.get('weights_sha256')==checkpoint and c.get('navigation_unchanged') is True and c.get('execution',{}).get('engine')=='merged_checkpoint_prefix' and c.get('execution',{}).get('prefix_layers')==5 and c.get('execution',{}).get('state_before')==c.get('execution',{}).get('state_after') and c.get('execution',{}).get('ticks_before')==c.get('execution',{}).get('ticks_after') for c in chats),
        'all_executions_in_checkpoint':bool(banks) and all(b.get('source_gate')=='trace_verified' and b.get('execution') and all(a.get('engine')=='merged_checkpoint_prefix' and a.get('prefix_layers')==5 and a.get('ticks_after',0)>a.get('ticks_before',0) for a in b['execution']) for b in banks),
    }
    proof['uncompleted_bank_executions']=proof['recorded_bank_executions']-proof['bank_executions']
    proof['uncompleted_triangle_installations']=proof['recorded_triangle_installations']-proof['resident_triangle_installations']
    proof['uncompleted_branch_commands']=proof['recorded_branch_commands']-proof['branch_commands']
    proof['branch_exercised']=proof['resident_triangle_installations']>0 and proof['branch_commands']>0
    proof['run_completed']=completed
    proof['passed']=completed and proof['completed_rounds_bound_to_banks'] and proof['all_completions_gated'] and proof['all_executions_in_checkpoint'] and (not require_branch or proof['branch_exercised'])
    return proof,chats,banks

class App:
    def __init__(self, out, heart, workspace, phase):
        self.out, self.phase = out, phase
        self.url=self.token=None
        self.lines=[]
        self.log=(out/f'{phase}.log').open('w')
        cmd=[str(BINARY),'--no-settings','chat',str(heart),'--workspace',str(workspace),
            '--model-dir',str(ROOT/'models/minilm'),'--nli-model-dir',str(ROOT/'models/nli-minilm'),
            '--no-download','--skip-onboarding','--web','--web-port','0','--thread','integration-full',
            '--temperature','0','--max-tokens','1024','--max-tool-rounds','4','--timeout-seconds','600',
            '--max-history-turns','2','--max-history-chars','5000',
            '--triangle-backend',str(ROOT/'.venv/bin/python')]
        for arg in [os.environ.get('SPINE_TEST_BACKEND_SCRIPT',str(FORK/'backends/triangle/server.py')),'--model-dir',str(ROOT/'models/qwen3.5-triangles'),
                    '--compiler-dir',str(ROOT/'vendor/qwenputer'),'--device','xpu']:
            cmd.append('--triangle-backend-arg='+arg)
        if os.environ.get('SPINE_TEST_ABLATE_RESIDENT') == '1':
            for arg in ['--runtime-dir',str(FORK/'backends/triangle'),'--ablate-resident-records']:
                cmd.append('--triangle-backend-arg='+arg)
        if os.environ.get('SPINE_TEST_TRIANGLE_CONTEXT_ONLY') == '1':
            cmd.append('--triangle-context-only')
        if os.environ.get('SPINE_TEST_TRIANGLE_INTERNAL_CONTEXT') == '1':
            cmd.append('--triangle-internal-context')
        self.child=subprocess.Popen(cmd,cwd=FORK,env=os.environ,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.reader=threading.Thread(target=self.read,daemon=True);self.reader.start()
        deadline=time.monotonic()+120
        while not self.url:
            if self.child.poll() is not None:raise RuntimeError(f'App exited {self.child.returncode}; see {phase}.log')
            if time.monotonic()>deadline:
                self.child.kill();self.child.wait();raise TimeoutError('App did not start')
            time.sleep(.2)
        assert self.api('state')['grounding'] is True
        print(phase,'started with grounding and normal tool registry',flush=True)
    def read(self):
        for line in self.child.stdout:
            safe_line=re.sub(r'(Spine web UI: http://[^/]+/#[^\s]+)', 'Spine web UI: [local authentication URL redacted]', line)
            self.lines.append(safe_line);self.log.write(safe_line);self.log.flush()
            match=re.search(r'Spine web UI: (http://[^/]+)/#(\w+)',line)
            if match:self.url,self.token=match.groups()
    def api(self,method,data=None):
        req=Request(self.url+'/api/'+method,headers={'x-spine-token':self.token,'Content-Type':'application/json'},
            data=json.dumps(data).encode() if data is not None else None)
        with urlopen(req,timeout=10) as r:
            body=r.read()
            return json.loads(body) if body else None
    def wait(self,label,timeout=600):
        started=time.monotonic();notice=started
        while True:
            state=self.api('state')
            if not state['busy']:
                (self.out/f'{self.phase}-{label}.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
                if state['phase']=='error':raise RuntimeError(f'App turn failed: {state.get("notice")}')
                return state
            now=time.monotonic()
            if now-started>timeout:raise TimeoutError(f'{label} exceeded {timeout}s')
            if now-notice>30:
                print(self.phase,label,'working:',state['activity'],round(now-started),'s',flush=True);notice=now
            time.sleep(.5)
    def ask(self,label,text):
        self.api('message',{'message':text})
        state=self.wait(label)
        answer=next(m['content'] for m in reversed(state['messages']) if m['role']=='assistant')
        return answer,state
    def close(self):
        if self.child.poll() is None:
            try:self.api('control',{'action':'quit'});self.child.wait(timeout=30)
            except Exception:self.child.terminate();self.child.wait(timeout=10)
        self.reader.join(timeout=2);self.log.close()
        assert self.child.returncode==0, f'App exit {self.child.returncode}'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--boundaries',action='store_true')
    parser.add_argument('--triangle-context-only',action='store_true')
    parser.add_argument('--triangle-internal-context',action='store_true',help='Require real compiled native-memory handoff in every completion')
    parser.add_argument('--require-triangle-branch',action='store_true',help='Fail navigation proof unless a real triangle branch executes')
    parser.add_argument('--query',action='append',help='Run only these retention query IDs; skip operational checks')
    parser.add_argument('--force-recall',action='store_true',help='Diagnostic only: explicitly request heart_recall before answering')
    args=parser.parse_args()
    if args.require_triangle_branch and not args.triangle_internal_context:
        parser.error('--require-triangle-branch requires --triangle-internal-context')
    if args.triangle_context_only:
        os.environ['SPINE_TEST_TRIANGLE_CONTEXT_ONLY']='1'
    if args.triangle_internal_context:
        os.environ['SPINE_TEST_TRIANGLE_INTERNAL_CONTEXT']='1'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    workspace=out/'workspace';workspace.mkdir()
    source=ROOT/('artifacts/retention/native-boundaries/retention.spine' if args.boundaries else 'artifacts/retention/native-v2/retention.spine')
    before=digest(source);heart=out/'test.spine';shutil.copy2(source,heart)
    os.environ['SPINE_HEART_PASSPHRASE']='synthetic-retention-test-only'
    os.environ['SPINE_DISABLE_AUTO_OPEN']='1'
    os.environ['SPINE_TRIANGLE_TRACE_DIR']=str(out/'model-responses')
    # Source-derived expected values and exact supporting event IDs, not model self-grading.
    corpus=json.loads((ROOT/('artifacts/retention/boundary-corpus.json' if args.boundaries else 'artifacts/retention/corpus.json')).read_text())
    native=json.loads((ROOT/('artifacts/retention/native-boundaries/boundary.json' if args.boundaries else 'artifacts/retention/native-v2/128k.json')).read_text())['native']
    labels={r['label']:r for r in native['event_labels']}
    checks=[];app=None
    report={'schema':1,'checkpoint':json.loads((ROOT/'models/qwen3.5-triangles/triangle.json').read_text())['weights_sha256'],
        'source_heart_sha256':before,'fixture_events':native['canonical_events'],'grounding_enabled':True,'checks':checks}
    report['automatic_memory_mode']='native_triangles_only' if args.triangle_context_only else 'native_default'
    report['compiled_internal_context']=args.triangle_internal_context
    report['binary_sha256']=digest(BINARY)
    report['require_triangle_branch']=args.require_triangle_branch
    report['expected_checks']=(len(args.query) if args.query else (1 if args.pilot else len(corpus['queries']))) + (0 if args.pilot or args.boundaries or args.query else 6)
    report['evaluation_mode']='explicit_recall_probe' if args.force_recall else 'unassisted_acceptance'
    def record(name,ok,**details):
        checks.append(dict(name=name,passed=bool(ok),**details))
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(name,'PASS' if ok else 'FAIL',flush=True)
    try:
        app=App(out,heart,workspace,'first')
        queries=corpus['queries'][:1] if args.pilot else corpus['queries']
        if args.query:
            missing=set(args.query)-{q['id'] for q in queries}
            if missing:raise ValueError(f'Unknown selected query IDs: {sorted(missing)}')
            queries=[q for q in queries if q['id'] in args.query]
        for q in queries:
            instruction=' Use existing heart evidence only. Do not search the web or filesystem. Cite the exact supporting event ID. Keep the answer to one sentence.'
            if args.force_recall:
                instruction+=' Before answering, call heart_recall with a query about this question. Inspect the original source text in the tool result; the initial triangle excerpts may be truncated or unrelated. If the tool result still does not support an answer, say the answer is unknown.'
            answer,state=app.ask(q['id'],q['question']+instruction)
            expected=corpus['stages'][-1]['current_expected'] if q['id']=='current' else q['expected']
            support=[labels[k]['event_id'] for k in q['support'] if k in labels]
            if q['id']=='current':support=[labels['deploy_new']['event_id']]
            passed=(expected in answer and any(i in answer for i in support)) if expected is not None else (
                not re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',answer) and any(x in answer.lower() for x in ['missing','not','unknown','unavailable','no ']))
            record('retention_'+q['id'],passed,answer=answer,expected=expected,support=support)
        if not (args.pilot or args.boundaries or args.query):
            (workspace/'tool-fixture.txt').write_text('The tool-only beacon is MARIGOLD-842.\n')
            answer,state=app.ask('tool-read','Call file_read to read tool-fixture.txt in this workspace. Report its exact beacon value in one sentence. This is a tool execution test; do not infer the file contents from memory.')
            record('real_tool_read','MARIGOLD-842' in answer and any(t['name']=='file_read' and t['success'] for t in state['tools']),answer=answer)
            answer,state=app.ask('tool-write','Use file_write to create integration-output.txt in this workspace with exactly the text SPINE-TRIANGLE-OK with no trailing newline. Then report completion. This write to the disposable test workspace is explicitly authorized.')
            target=workspace/'integration-output.txt'
            record('real_tool_write',target.exists() and target.read_text()=='SPINE-TRIANGLE-OK',answer=answer)
            answer,state=app.ask('remember','Remember this new session fact: the local restart beacon is COMET-581. Acknowledge briefly without using tools.')
            record('session_fact_acknowledged',state['phase']=='idle',answer=answer)
            app.close();app=None
            app=App(out,heart,workspace,'reopened')
            answer,state=app.ask('restart','What was the local restart beacon I gave you in our previous session? Return the exact value and its source event ID. Use only heart memory, not the filesystem.')
            record('new_process_heart_recall','COMET-581' in answer and bool(re.search('[a-f0-9]{64}',answer)),answer=answer)
            # Interrupt an actual request. Dropping the in-flight provider must kill the model
            # child, keep the journal recoverable and allow a new process on the next turn.
            app.api('message',{'message':'Write a very long detailed essay comparing all the Zircon records, at least 3000 words. Do not use tools.'})
            time.sleep(2)
            app.api('control',{'action':'interrupt'})
            state=app.wait('interrupted')
            record('operator_interrupt',not state['busy'],phase=state['phase'],checkpoint=state['checkpoint_available'])
            answer,state=app.ask('after-interrupt','What is the local restart beacon? Cite its event ID. Answer briefly using heart memory only.')
            record('inference_after_interrupt','COMET-581' in answer,answer=answer)
    except BaseException as error:
        report['run_error']=str(error)
        raise
    finally:
        if app is not None:app.close()
        report['source_heart_unchanged']=digest(source)==before
        report['run_completed']=len(checks)==report['expected_checks']
        report['behavioral_all_passed']=report['run_completed'] and all(c['passed'] for c in checks) and report['source_heart_unchanged']
        report['all_passed']=report['behavioral_all_passed']
        if args.triangle_internal_context:
            events=read_navigation_events(out.glob('*.log'))
            proof,chats,banks=navigation_evidence(events,report['checkpoint'],report['run_completed'],args.require_triangle_branch)
            report['internal_navigation']=proof
            report['all_passed']=report['all_passed'] and proof['passed']
            (out/'internal-navigation-events.json').write_text(json.dumps({'chats':chats,'banks':banks},ensure_ascii=False,indent=2)+'\n')
            print('Internal navigation:',json.dumps(proof),flush=True)
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not report['all_passed']:raise SystemExit('Acceptance checks failed; evidence retained')

if __name__=='__main__':main()
