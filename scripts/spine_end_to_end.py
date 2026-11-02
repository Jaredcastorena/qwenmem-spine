"""Native browser/agent acceptance; run only through scripts/guarded-python.sh.

Uses a disposable copy of the synthetic 128K heart, never a personal heart.
The public chat API, model tools, journal, grounding and projection remain enabled.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path
from urllib.request import Request, urlopen
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'artifacts/spine-target/debug/spine'
FORK=ROOT/'rust-spine'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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
        for arg in [str(FORK/'backends/triangle/server.py'),'--model-dir',str(ROOT/'models/qwen3.5-triangles'),
                    '--compiler-dir',str(ROOT/'vendor/qwenputer'),'--device','xpu']:
            cmd.append('--triangle-backend-arg='+arg)
        if os.environ.get('SPINE_TEST_TRIANGLE_CONTEXT_ONLY') == '1':
            cmd.append('--triangle-context-only')
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
    parser.add_argument('--query',action='append',help='Run only these retention query IDs; skip operational checks')
    parser.add_argument('--force-recall',action='store_true',help='Diagnostic only: explicitly request heart_recall before answering')
    args=parser.parse_args()
    if args.triangle_context_only:
        os.environ['SPINE_TEST_TRIANGLE_CONTEXT_ONLY']='1'
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
        if args.pilot or args.boundaries or args.query:
            if not all(c['passed'] for c in checks):raise SystemExit('Acceptance failed; evidence retained')
            return
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
    finally:
        if app is not None:app.close()
        report['source_heart_unchanged']=digest(source)==before
        report['all_passed']=all(c['passed'] for c in checks) and report['source_heart_unchanged']
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not report['all_passed']:raise SystemExit('Acceptance checks failed; evidence retained')

if __name__=='__main__':main()
