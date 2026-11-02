"""Local triangle-memory experiments. No production-heart access or network calls."""
import argparse
import json
from pathlib import Path
import time

from . import protocol as p
from .forest import Forest
from .hf_runner import ModelRunner


def collect(forest,trace,remaining=4096):
    evidence=[]
    for token in trace:
        if p.APEX <= token < p.APEX+p.CAPACITY:
            item=forest.evidence(token-p.APEX,max_bytes=remaining)
            remaining-=len(item["text"].encode())
            evidence.append(item)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest",default="examples/synthetic-forest.json")
    parser.add_argument("--device",choices=["cpu","xpu"],default="cpu")
    parser.add_argument("--model",help="Merged Qwen checkpoint; omit for standalone compiled controller")
    subs=parser.add_subparsers(dest="action",required=True)
    run=subs.add_parser("run",help="Run commands; optionally restore/save persistent triangle state")
    run.add_argument("commands",nargs="+",help="open:SLOT, peek, left, right, route:INDEX")
    run.add_argument("--restore")
    run.add_argument("--save")
    run.add_argument("--ask",help="Question to answer from the retrieved evidence (requires --model)")
    demo=subs.add_parser("demo",help="Saved alternate route, fresh session, exception recall")
    demo.add_argument("--output",default="artifacts/demo")
    args=parser.parse_args()
    if getattr(args,"ask",None) and not args.model:
        parser.error("--ask requires --model")
    forest=Forest.load(args.forest)
    start=time.monotonic()
    if args.model:
        from .merged_runner import MergedRunner
        runner=MergedRunner(forest,device=args.device,model_path=args.model)
    else:
        runner=ModelRunner(forest,device=args.device)
    report={"device":args.device,"forest_sha256":forest.digest,"weights_sha256":runner.weight_id,
            "load_seconds":round(time.monotonic()-start,3),"traces":[]}
    evidence=[]
    def execute(token):
        before=time.monotonic()
        trace=runner.execute(token)
        report["traces"].append({"tokens":[p.name(t) for t in trace],
                                 "seconds":round(time.monotonic()-before,4)})
        print(" → ".join(p.name(t) for t in trace),flush=True)
        return trace
    if args.action=="run":
        if args.restore:
            runner.restore(args.restore)
        for cmd in args.commands:
            evidence.extend(collect(forest,execute(p.command(cmd)),4096-sum(len(e["text"].encode()) for e in evidence)))
        if args.save:
            runner.save(args.save)
        question=args.ask
    else:
        out=Path(args.output)
        out.mkdir(parents=True,exist_ok=True)
        root=forest.slots[forest.roots[0]]
        execute(root)
        execute(p.RIGHT)
        execute(p.RIGHT)
        snapshot=out/"session.safetensors"
        runner.save(snapshot)
        if args.model:
            runner.reset_session()
        else:
            del runner
            runner=ModelRunner(forest,device=args.device)
        runner.restore(snapshot)
        # The earlier decision branch is reached from saved route 0, then its exception.
        execute(p.ROUTE)
        evidence.extend(collect(forest,execute(p.RIGHT)))
        question="Which port must the offline recovery endpoint use, and does changing the main port change it?"
        report.update(snapshot_bytes=snapshot.stat().st_size,previous_transcript_tokens=0)
    report["evidence"]=evidence
    if question and args.model:
        before=time.monotonic()
        report["language"]=runner.answer(question,evidence)
        report["language"]["seconds"]=round(time.monotonic()-before,3)
        print(report["language"]["answer"],flush=True)
    report["total_seconds"]=round(time.monotonic()-start,3)
    if args.action=="demo":
        (Path(args.output)/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
