"""Prepend compiled triangle layers to local Qwen3.5 language weights.

Derived from Qwenputer's MIT-licensed rv/merge.py (see THIRD_PARTY.md).
The runtime dispatches triangle tokens to the prefix and text to the language stack.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

from . import circuit, protocol as p


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def build(src, dst):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from chipformer.q35.emit import Layout, emit_layers, embedding_rows
    from chipformer.q35.merge import IN_COLS, OUT_ROWS, WIDE_NORMS, pick_hidden
    from chipformer.q35.ir import Target

    torch.set_num_threads(4)
    src, dst = Path(src), Path(dst)
    if dst.exists() and any(dst.iterdir()):
        raise ValueError("Choose an empty output directory")
    original = json.loads((src/"config.json").read_text())["text_config"]
    d0, v0 = original["hidden_size"], original["vocab_size"]
    c = circuit.build(Target(original["linear_num_key_heads"],
                             original["linear_num_value_heads"], original["intermediate_size"]))
    hidden, m = pick_hidden(d0, len(c.dims))
    layout = Layout(hidden=hidden, off=d0, m=m)
    assert layout.dim(len(c.dims)-1) < hidden
    n = c.n_layers()
    cfg = {k:v for k,v in original.items() if k != "full_attention_interval"}
    cfg.update(hidden_size=hidden, num_hidden_layers=n+original["num_hidden_layers"],
               layer_types=["linear_attention"]*n+original["layer_types"],
               mtp_num_hidden_layers=0, architectures=["Qwen3_5ForCausalLM"],
               model_type="qwen3_5_text", vocab_size=v0+p.VOCAB_SIZE,
               tie_word_embeddings=True)
    print(json.dumps({"circuit":c.stats(),"hidden":hidden,"ballast":m}),flush=True)
    sd, base_emb = {}, None
    prefix = "model.language_model."
    for file in sorted(src.glob("*.safetensors")):
        with safe_open(file,framework="pt") as f:
            for name in f.keys():
                if not name.startswith(prefix):
                    continue
                w = f.get_tensor(name)
                if name == prefix+"embed_tokens.weight":
                    base_emb = w
                    continue
                if IN_COLS.search(name):
                    w = torch.cat([w,w.new_zeros(*w.shape[:-1],hidden-d0)],-1)
                elif OUT_ROWS.search(name):
                    w = torch.cat([w,w.new_zeros(hidden-d0,*w.shape[1:])],0)
                elif WIDE_NORMS.search(name):
                    gain = ((1+w.float())*(d0/hidden)**0.5-1).to(w.dtype)
                    w = torch.cat([gain,w.new_full((hidden-d0,),-1)])
                dest = "model."+name[len(prefix):]
                dest = re.sub(r"^(model\.layers\.)(\d+)\.",
                              lambda match:f"{match[1]}{int(match[2])+n}.",dest)
                sd[dest] = w
    if base_emb is None:
        raise ValueError("Expected Qwen3.5 multimodal source checkpoint")
    print("Compiling triangle prefix",flush=True)
    layers = emit_layers(c,layout,cfg,0,"model.",keep=c.keep)
    for key in list(layers):
        sd[key] = layers.pop(key).to(torch.bfloat16)
    emb = torch.zeros(cfg["vocab_size"],hidden,dtype=torch.bfloat16)
    emb[:v0,:d0] = base_emb
    emb[v0:] = embedding_rows(c,layout,list(range(p.VOCAB_SIZE)),hidden).to(torch.bfloat16)
    sd["model.embed_tokens.weight"] = emb
    dst.mkdir(parents=True,exist_ok=True)
    shards, current, size = [], {}, 0
    for key in sorted(sd):
        current[key] = sd[key].contiguous()
        size += sd[key].numel()*sd[key].element_size()
        if size >= 2**30:
            shards.append(current)
            current, size = {}, 0
    if current:
        shards.append(current)
    index = {"metadata":{"total_size":sum(t.numel()*t.element_size() for t in sd.values())},
             "weight_map":{}}
    files = {}
    for i,shard in enumerate(shards):
        name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        print(f"Saving {name}",flush=True)
        save_file(shard,str(dst/name),metadata={"format":"pt"})
        files[name] = file_digest(dst/name)
        index["weight_map"].update({key:name for key in shard})
    (dst/"model.safetensors.index.json").write_text(json.dumps(index,indent=2)+"\n")
    (dst/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    for name in ("tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE"):
        if (src/name).exists():
            shutil.copyfile(src/name,dst/name)
    meta = {"schema":1,"chip_layers":n,"text_vocab":v0,"original_hidden":d0,
            "code_dims":[layout.dim(next(iter(e.terms))) for e in c.code],
            "circuit":c.stats(),"files":files,
            "dispatch":"triangle prefix for commands; language layers for text",
            "source_config_sha256":file_digest(src/"config.json")}
    meta["weights_sha256"] = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    (dst/"triangle.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta),flush=True)
    return meta


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args=parser.parse_args()
    build(args.source,args.output)
