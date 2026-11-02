"""Actual normal terminal-path check using a copied, synthetic completed heart."""
import argparse,json,os,re,shutil,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--heart',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);heart=out/'terminal.spine';shutil.copy2(a.heart,heart)
 os.environ['SPINE_HEART_PASSPHRASE']='synthetic-retention-test-only'
 cmd=[str(ROOT/'artifacts/spine-target/debug/spine'),'--no-settings','chat',str(heart),'--workspace',str(out),
      '--model-dir',str(ROOT/'models/minilm'),'--nli-model-dir',str(ROOT/'models/nli-minilm'),'--no-download',
      '--skip-onboarding','--terminal','--thread','terminal-smoke','--temperature','0','--max-tokens','512',
      '--triangle-backend',str(ROOT/'.venv/bin/python')]
 for arg in [str(ROOT/'rust-spine/backends/triangle/server.py'),'--model-dir',str(ROOT/'models/qwen3.5-triangles'),
             '--compiler-dir',str(ROOT/'vendor/qwenputer'),'--device','xpu']:cmd+=['--triangle-backend-arg='+arg]
 child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 lines=[];ready=threading.Event();answered=threading.Event()
 def read():
  with (out/'terminal.log').open('w') as log:
   for line in child.stdout:
    lines.append(line);log.write(line);log.flush()
    if 'Spine ready:' in line:ready.set()
    if 'spine>' in line and 'COMET-581' in line:answered.set()
 reader=threading.Thread(target=read,daemon=True);reader.start()
 try:
  if not ready.wait(90):raise RuntimeError('Terminal not ready')
  child.stdin.write('What was the local restart beacon from the previous session? Use heart evidence and cite its event ID. Answer in one sentence.\n');child.stdin.flush()
  if not answered.wait(240):raise RuntimeError('No correct terminal answer before deadline')
  child.stdin.write('/quit\n');child.stdin.flush();child.wait(timeout=30)
 finally:
  if child.poll() is None:child.kill();child.wait()
  reader.join(timeout=3)
 answers=[line.strip() for line in lines if 'spine>' in line and 'COMET-581' in line]
 result={'passed':child.returncode==0 and bool(answers) and any(re.search('[a-f0-9]{64}',a) for a in answers),
         'exit_code':child.returncode,'answers':answers,'grounding_enabled':True}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
