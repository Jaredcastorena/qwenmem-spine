"""Exact full-history control with fixed KV storage and resumable layer shards.

Run through scripts/guarded-python.sh. The legacy raw runner remains capped at 45K tokens.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from triangle_memory.bounded_cache import BoundedKV,make_cache,prompt_digest,save_checkpoint,restore_checkpoint
from triangle_memory.forest import Forest
from triangle_memory.merged_runner import MergedRunner
from triangle_memory.raw_audit import control_inputs,audit_answer


def host_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def memory():
    return dict(host_available_bytes=host_available(),device_allocated_bytes=torch.xpu.memory_allocated(),
                device_reserved_bytes=torch.xpu.memory_reserved(),device_peak_allocated_bytes=torch.xpu.max_memory_allocated(),
                device_peak_reserved_bytes=torch.xpu.max_memory_reserved())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['4k','32k','128k'],required=True)
    parser.add_argument('--output',default='artifacts/retention/full-bounded')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=65536)
    parser.add_argument('--check-numerics',action='store_true',default=True,
                        help='Validate every delta-rule output and capture the first failure (always enabled)')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result_file=output/f'{args.stage}.json'
    if result_file.exists(): raise FileExistsError(f'Preserving completed result: {result_file}')
    # Enforce the same host control even if someone invokes this script directly.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from resource_guard import checked_cgroup
    checked_cgroup()
    torch.xpu.set_per_process_memory_fraction(0.70)
    metadata,_,_,_,messages=control_inputs(args.stage)
    runner=MergedRunner(Forest.load('examples/synthetic-forest.json'),model_path='models/qwen3.5-triangles',device='xpu')
    runner.model.set_attn_implementation('sdpa')
    runner.execute(6);before=runner._state()
    if args.check_numerics:
        from triangle_memory.numerical_probe import install_probe
        install_probe(runner.model,output/'numerical-probe')
    ids=runner.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if not isinstance(ids,list): ids=ids['input_ids']
    capacity=len(ids)+1400
    if capacity>runner.model.config.max_position_embeddings: raise ValueError('Configured model context exceeded')
    if len(ids)>180000: raise ValueError('Bounded experiment limited to 180K prompt tokens')
    stamp=time.time_ns();log=output/f'{args.stage}-{stamp}.progress.jsonl'
    start=time.monotonic();measurements=[]
    def record(phase,position,**extra):
        row=dict(phase=phase,position=position,prompt_tokens=len(ids),seconds=round(time.monotonic()-start,3),**memory(),**extra)
        measurements.append(row)
        with log.open('a') as file: file.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    if args.resume:
        cache,logits,position=restore_checkpoint(args.resume,runner.model.config,ids=ids,capacity=capacity,
                                                weights=runner.weight_id,device=runner.device)
        record('restored',position,checkpoint=str(args.resume))
    else:
        cache=make_cache(runner.model.config,capacity);position=0;logits=None
        record('ready',0,prompt_sha256=prompt_digest(ids),capacity=capacity,
               estimated_full_attention_cache_bytes=8*2*4*capacity*256*2)
    if args.stop_after and not position<args.stop_after<=len(ids): raise ValueError('Stop point must advance within prompt')
    def checkpoint():
        path=output/f'{args.stage}-cache-{position}'
        if path.exists() or path.with_name(path.name+'.partial').exists():
            path=path.with_name(path.name+f'-{time.time_ns()}')
        save_checkpoint(path,cache,logits,ids=ids,position=position,capacity=capacity,weights=runner.weight_id)
        record('checkpoint',position,checkpoint=str(path))
    with torch.inference_mode(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.OVERRIDEABLE]):
        while position<len(ids):
            end=min(position+256,len(ids),args.stop_after or len(ids))
            result=runner.model(torch.tensor([ids[position:end]],device=runner.device),
                                past_key_values=cache,use_cache=True,logits_to_keep=1)
            assert result.past_key_values is cache
            logits=result.logits[0,-1];position=end
            # Bound device command queues and host-side in-flight allocations.
            torch.xpu.synchronize()
            if not torch.isfinite(logits).all().item():
                record('numerical_failure',position)
                raise FloatingPointError('Non-finite prefill logits; no answer may be scored as valid')
            if host_available()<16*2**30: raise RuntimeError('Host headroom reserve reached')
            if torch.xpu.memory_allocated()>22*2**30: raise RuntimeError('Live device allocation reserve reached')
            if position%2048==0 or position==len(ids) or position==args.stop_after:
                record('prefill',position)
                torch.xpu.empty_cache()
            if position==args.stop_after or (args.checkpoint_every and position%args.checkpoint_every==0): checkpoint()
            if position==args.stop_after:
                record('intentional_stop',position)
                return
        retained_at_prefill={str(i):layer.length for i,layer in enumerate(cache.layers) if isinstance(layer,BoundedKV)}
        assert retained_at_prefill and all(n==len(ids) for n in retained_at_prefill.values())
        if args.stage=='128k' and not (args.resume and position==json.loads((args.resume/'manifest.json').read_text())['position']):
            checkpoint()
        generated=[]
        for n in range(1400):
            if not torch.isfinite(logits).all().item():
                record('numerical_failure',len(ids)+n)
                raise FloatingPointError('Non-finite decode logits; generation stopped')
            token=int(logits[:runner.meta['text_vocab']].argmax())
            if token in {runner.tokenizer.eos_token_id,248044,248046}: break
            generated.append(token)
            logits=runner._text_step(token,cache,len(ids)+n,final=True)
            torch.xpu.synchronize()
            if n%128==0:
                record('decode',len(ids)+n,answer_tokens=len(generated))
                partial=dict(complete=False,stage=args.stage,prompt_sha256=prompt_digest(ids),generated_ids=generated,
                             answer=runner.tokenizer.decode(generated,skip_special_tokens=True))
                (output/f'{args.stage}.answer.partial.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2)+'\n')
        answer=runner.tokenizer.decode(generated,skip_special_tokens=True)
    record('completed',position,answer_tokens=len(generated))
    report=dict(stage=metadata,answer=answer,prompt_tokens=len(ids),prompt_sha256=prompt_digest(ids),
                answer_tokens=len(generated),generated_ids=generated,truncated=len(generated)==1400,
                seconds_this_process=round(time.monotonic()-start,3),resume=str(args.resume) if args.resume else None,
                backend='fixed-capacity KV; fused-only SDPA; 256-token synchronized prefill; no token eviction',
                numerical_validation='every delta-rule output plus each prefill/decode logit vector',
                full_attention_tokens_at_prefill=retained_at_prefill,
                triangle_state_unchanged=all(torch.equal(v,runner._state()[k]) for k,v in before.items()),
                telemetry=measurements)
    # Raw outputs survive a failed verifier or an unexpected JSON layout.
    result_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['audit']=audit_answer(answer,args.stage)
    previous=Path(f'artifacts/retention/full-sdpa/{args.stage}.json')
    if previous.exists():
        baseline=json.loads(previous.read_text())
        report['previous_raw_answer_byte_equal']=answer==baseline['answer']
        report['previous_raw_prompt_tokens_equal']=len(ids)==baseline['prompt_tokens']
    result_file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('VERIFIED',report['audit']['verified'],'/',report['audit']['total'],flush=True)
    if report['audit']['verified']!=report['audit']['total'] or report['truncated']:
        raise RuntimeError('Completed raw output did not pass independent audit; output retained')


if __name__=='__main__': main()
