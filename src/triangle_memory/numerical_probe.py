"""Capture the first non-finite delta-rule operation for a small CPU/XPU replay."""
import inspect
import json
from pathlib import Path
import torch
from safetensors.torch import save_file


def install_probe(model,directory):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    original=module.torch_chunk_gated_delta_rule
    signature=inspect.signature(original)
    current={'layer':None}
    for layer in model.model.layers:
        if hasattr(layer,'linear_attn'):
            layer.linear_attn.register_forward_pre_hook(lambda m,args:current.update(layer=m.layer_idx))
    def checked(*args,**kwargs):
        output=original(*args,**kwargs)
        if all(t is None or torch.isfinite(t).all().item() for t in output): return output
        values=signature.bind(*args,**kwargs);values.apply_defaults()
        tensors={k:v.detach().cpu().contiguous() for k,v in values.arguments.items() if isinstance(v,torch.Tensor)}
        options={k:v for k,v in values.arguments.items() if not isinstance(v,torch.Tensor)}
        options.update(options.pop('kwargs',{}))
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        save_file(tensors,str(directory/'delta-inputs.safetensors'))
        report=dict(layer=current['layer'],options=options,
                    inputs_finite={k:bool(torch.isfinite(v).all()) for k,v in tensors.items()},
                    xpu_output_finite=[bool(torch.isfinite(t).all()) if t is not None else None for t in output])
        cpu=original(**{**options,**tensors})
        report['cpu_output_finite']=[bool(torch.isfinite(t).all()) if t is not None else None for t in cpu]
        report['cpu_max_abs']=[float(t.abs().max()) if t is not None else None for t in cpu]
        (directory/'delta-report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise FloatingPointError(f'Non-finite delta rule at layer {current["layer"]}; captured in {directory}')
    module.torch_chunk_gated_delta_rule=checked
