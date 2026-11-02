"""Append-only KV cache with fixed capacity and layer-sharded local checkpoints.

No token eviction, quantization, summarization, or reset at a shard boundary.
Only the batch-one, greedy, hybrid Qwen3.5 control is supported.
"""
import hashlib
import json
import os
from pathlib import Path
import struct

import torch
from safetensors.torch import load_file, save_file
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
from .merge import file_digest


class BoundedKV(DynamicLayer):
    def __init__(self, capacity):
        super().__init__()
        self.capacity=capacity
        self.length=0

    def lazy_initialization(self, keys, values):
        self.dtype,self.device=keys.dtype,keys.device
        self.keys=torch.empty((*keys.shape[:-2],self.capacity,keys.shape[-1]),dtype=keys.dtype,device=keys.device)
        self.values=torch.empty((*values.shape[:-2],self.capacity,values.shape[-1]),dtype=values.dtype,device=values.device)
        self.is_initialized=True

    def update(self, keys, values, *args, **kwargs):
        end=self.length+keys.shape[-2]
        if end>self.capacity: raise ValueError('Fixed KV capacity exceeded')
        if not self.is_initialized: self.lazy_initialization(keys,values)
        self.keys[...,self.length:end,:].copy_(keys)
        self.values[...,self.length:end,:].copy_(values)
        self.length=end
        return self.keys[...,:end,:],self.values[...,:end,:]

    def get_seq_length(self): return self.length
    def get_max_length(self): return self.capacity
    def crop(self,*args): raise NotImplementedError('Append-only control cache')
    def reset(self):
        self.keys=self.values=None
        self.is_initialized=False
        self.length=0


def make_cache(config,capacity):
    cache=DynamicCache(config=config)
    for i,layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer: cache.layers[i]=BoundedKV(capacity)
        elif type(layer) is not LinearAttentionLayer: raise TypeError(f'Unsupported cache layer: {type(layer)}')
    return cache


def prompt_digest(ids):
    return hashlib.sha256(struct.pack(f'<{len(ids)}I',*ids)).hexdigest()


def host_tensor(tensor):
    # Copy bounded slabs, without registering a whole mapped cache on the GPU.
    result=torch.empty(tensor.shape,dtype=tensor.dtype,device='cpu')
    if tensor.ndim>=3:
        for start in range(0,tensor.shape[-2],512):
            result[...,start:start+512,:].copy_(tensor[...,start:start+512,:])
    else: result.copy_(tensor)
    return result.contiguous()


def durable(path):
    descriptor=os.open(path,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def save_checkpoint(path,cache,logits,*,ids,position,capacity,weights):
    if not torch.isfinite(logits).all().item(): raise FloatingPointError('Refusing checkpoint with non-finite logits')
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    temporary=path.with_name(path.name+'.partial')
    if temporary.exists(): raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    manifest=dict(schema=1,prompt_sha256=prompt_digest(ids),position=position,capacity=capacity,
                  weights=weights,prompt_tokens=len(ids),files={})
    try:
        for i,layer in enumerate(cache.layers):
            if isinstance(layer,BoundedKV):
                assert layer.length==position
                tensors={'keys':host_tensor(layer.keys[...,:position,:]),'values':host_tensor(layer.values[...,:position,:])}
            else:
                tensors={f'{kind}.{j}':host_tensor(t) for kind in ('conv_states','recurrent_states')
                         for j,t in getattr(layer,kind).items() if t is not None}
            file=temporary/f'layer-{i:02d}.safetensors'
            save_file(tensors,str(file));del tensors
            manifest['files'][file.name]=file_digest(file)
            durable(file)
        file=temporary/'logits.safetensors'
        save_file({'logits':host_tensor(logits)},str(file))
        manifest['files'][file.name]=file_digest(file)
        durable(file)
        manifest_file=temporary/'manifest.json'
        manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
        durable(manifest_file);durable(temporary)
        temporary.rename(path)
        durable(path.parent)
    except BaseException:
        # Incomplete shards are never offered as a resumable checkpoint.
        raise
    return manifest


def restore_checkpoint(path,config,*,ids,capacity,weights,device):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    expected=dict(schema=1,prompt_sha256=prompt_digest(ids),capacity=capacity,weights=weights,prompt_tokens=len(ids))
    if any(manifest.get(k)!=v for k,v in expected.items()): raise ValueError('Checkpoint identity mismatch')
    position=manifest['position']
    if not 0<position<=len(ids): raise ValueError('Invalid checkpoint position')
    names={f'layer-{i:02d}.safetensors' for i in range(config.num_hidden_layers)}|{'logits.safetensors'}
    if set(manifest['files'])!=names: raise ValueError('Unexpected checkpoint shards')
    # Verify every file before creating a live GPU cache.
    for name,digest in manifest['files'].items():
        if file_digest(path/name)!=digest: raise ValueError(f'Corrupt checkpoint shard: {name}')
    logits=load_file(str(path/'logits.safetensors'),device='cpu')['logits']
    if not torch.isfinite(logits).all().item(): raise FloatingPointError('Checkpoint contains non-finite logits')
    logits=logits.clone().to(device)
    cache=make_cache(config,capacity)
    for i,layer in enumerate(cache.layers):
        tensors=load_file(str(path/f'layer-{i:02d}.safetensors'),device='cpu')
        if isinstance(layer,BoundedKV):
            keys,values=tensors['keys'],tensors['values']
            shape=(1,config.num_key_value_heads,position,config.head_dim)
            if keys.shape!=shape or values.shape!=shape: raise ValueError('Invalid KV shape')
            for start in range(0,position,512):
                layer.update(keys[...,start:start+512,:].clone().to(device),values[...,start:start+512,:].clone().to(device))
        else:
            for name,tensor in tensors.items():
                kind,index=name.split('.')
                if kind not in ('conv_states','recurrent_states') or index!='0': raise ValueError('Invalid recurrent state key')
                value=tensor.clone().to(device)
                if kind=='conv_states': layer.update_conv_state(value)
                else: layer.update_recurrent_state(value)
        del tensors
    return cache,logits,position
