import torch
import pytest
from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,DynamicCache
from triangle_memory.bounded_cache import BoundedKV,make_cache,save_checkpoint,restore_checkpoint


def test_bounded_cache_preserves_every_key_and_rejects_overflow():
    layer=BoundedKV(7)
    first=torch.arange(12).reshape(1,2,3,2).float()
    second=first+100
    layer.update(first,first)
    address=layer.keys.data_ptr()
    keys,_=layer.update(second,second)
    assert torch.equal(keys,torch.cat([first,second],dim=-2))
    assert layer.keys.data_ptr()==address
    assert layer.get_mask_sizes(1)==(7,0)
    with pytest.raises(ValueError): layer.update(second,second)
    assert layer.length==6


def test_sharded_checkpoint_restores_kv_and_recurrence_and_rejects_wrong_prompt(tmp_path):
    cfg=Qwen3_5TextConfig(num_hidden_layers=2,layer_types=['linear_attention','full_attention'],num_key_value_heads=2,head_dim=4)
    cache=make_cache(cfg,8)
    keys=torch.arange(24).reshape(1,2,3,4).to(torch.bfloat16)
    cache.layers[1].update(keys,keys+1)
    cache.layers[0].update_conv_state(torch.randn(1,16,4))
    cache.layers[0].update_recurrent_state(torch.randn(1,2,4,4))
    logits=torch.randn(12)
    path=tmp_path/'state'
    save_checkpoint(path,cache,logits,ids=[1,2,3,4],position=3,capacity=8,weights='test')
    restored,output,position=restore_checkpoint(path,cfg,ids=[1,2,3,4],capacity=8,weights='test',device='cpu')
    assert position==3 and torch.equal(output,logits)
    assert torch.equal(restored.layers[1].keys[...,:3,:],keys)
    assert torch.equal(restored.layers[0].recurrent_states[0],cache.layers[0].recurrent_states[0])
    assert torch.equal(restored.layers[0].conv_states[0],cache.layers[0].conv_states[0])
    assert restored.layers[0].has_previous_state[0]
    with pytest.raises(ValueError,match='identity'):
        restore_checkpoint(path,cfg,ids=[1,2,3,5],capacity=8,weights='test',device='cpu')
    with (path/'layer-01.safetensors').open('ab') as file: file.write(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):
        restore_checkpoint(path,cfg,ids=[1,2,3,4],capacity=8,weights='test',device='cpu')


def test_hybrid_model_logits_match_dynamic_cache_after_restart(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(4)
    cfg=Qwen3_5TextConfig(vocab_size=64,hidden_size=128,intermediate_size=256,
        num_hidden_layers=4,layer_types=['linear_attention','full_attention']*2,
        num_attention_heads=2,num_key_value_heads=2,head_dim=64,
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
        rope_parameters={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':0.25,'mrope_section':[2,3,3]})
    model=Qwen3_5ForCausalLM(cfg).eval()
    model.set_attn_implementation('sdpa')
    ids=list(range(1,29))
    dynamic=DynamicCache(config=cfg);bounded=make_cache(cfg,64)
    with torch.inference_mode():
        for start in range(0,28,7):
            chunk=torch.tensor([ids[start:start+7]])
            expected=model(chunk,past_key_values=dynamic,use_cache=True).logits
            actual=model(chunk,past_key_values=bounded,use_cache=True).logits
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
            if start==7:
                path=tmp_path/'restart'
                save_checkpoint(path,bounded,actual[0,-1],ids=ids,position=14,capacity=64,weights='fixture')
                bounded,_,_=restore_checkpoint(path,cfg,ids=ids,capacity=64,weights='fixture',device='cpu')


def test_nonfinite_checkpoint_is_rejected_without_publishing(tmp_path):
    cfg=Qwen3_5TextConfig(num_hidden_layers=1,layer_types=['full_attention'])
    with pytest.raises(FloatingPointError,match='non-finite'):
        save_checkpoint(tmp_path/'invalid',make_cache(cfg,8),torch.tensor([float('nan')]),
                        ids=[1],position=1,capacity=8,weights='fixture')
    assert not (tmp_path/'invalid').exists()
