"""Bound token-independent gated MLP intermediates without truncating context."""
from functools import wraps


def enable_mlp_chunking(model,chunk_size):
    import torch
    from torch.utils.checkpoint import checkpoint
    if chunk_size<1:raise ValueError('MLP chunk size must be positive')
    count=0
    for module in model.modules():
        if not all(hasattr(module,name) for name in ('gate_proj','up_proj','down_proj')):continue
        if getattr(module,'_sft_mlp_chunk_size',None)==chunk_size:continue
        if hasattr(module,'_sft_mlp_chunk_size'):raise ValueError('MLP chunking is already enabled with another size')
        original=module.forward
        @wraps(original)
        def forward(x,_forward=original):
            parts=[]
            for start in range(0,x.shape[-2],chunk_size):
                part=x[...,start:start+chunk_size,:]
                parts.append(checkpoint(_forward,part,use_reentrant=False) if torch.is_grad_enabled() else _forward(part))
            return torch.cat(parts,dim=-2)
        module.forward=forward;module._sft_mlp_chunk_size=chunk_size;count+=1
    if not count:raise ValueError('No compatible gated MLP modules found')
    return count
