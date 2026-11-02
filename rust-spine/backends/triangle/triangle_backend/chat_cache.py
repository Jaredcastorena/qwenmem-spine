"""Fixed-capacity language cache; no eviction or hidden context truncation."""
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

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


