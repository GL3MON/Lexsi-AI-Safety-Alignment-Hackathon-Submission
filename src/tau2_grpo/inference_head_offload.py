"""Small, inference-only CUDA workspace for a CPU-resident output projection."""
import torch
from torch import nn


class CudaChunkedHead(nn.Module):
    """Keep weights on CPU and stream vocabulary chunks through bounded VRAM."""

    def __init__(self, original, chunk_rows=8192):
        super().__init__()
        if original.weight.device.type != 'cpu':
            raise ValueError('Output projection must remain CPU resident')
        self.weight = original.weight
        self.bias = getattr(original, 'bias', None)
        self.chunk_rows = chunk_rows

    def forward(self, hidden):
        if torch.is_grad_enabled():
            raise RuntimeError('CUDA head offload is inference only')
        destination = hidden.device
        gpu_hidden = hidden.to('cuda:0')
        outputs = []
        for start in range(0, self.weight.shape[0], self.chunk_rows):
            end = start + self.chunk_rows
            weight = self.weight[start:end].to('cuda:0')
            bias = self.bias[start:end].to('cuda:0') if self.bias is not None else None
            outputs.append(torch.nn.functional.linear(gpu_hidden, weight, bias).to(destination))
            del weight, bias
        return torch.cat(outputs, dim=-1)


def enable_bounded_cuda_head(policy, memory_mib=256):
    """Cap this process's allocator; never move the full model or KV cache to GPU."""
    if policy.cfg.device != 'cpu' or memory_mib < 128:
        raise ValueError('Bounded head requires a CPU policy and at least 128 MiB')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one inference GPU')
    free, total = torch.cuda.mem_get_info()
    limit = memory_mib * 2**20
    # Keep two GiB plus the requested workspace free for the existing trainer.
    if free < limit + 2 * 2**30:
        raise RuntimeError('Insufficient free memory to preserve training headroom')
    torch.cuda.set_per_process_memory_fraction(limit / total, device=0)
    base = policy.model.get_base_model()
    base.lm_head = CudaChunkedHead(base.lm_head, chunk_rows=32768 if memory_mib >= 1024 else 8192)
    return {'mode': 'cpu_model_cuda_chunked_head', 'allocator_limit_mib': memory_mib,
            'weight_chunk_mib': base.lm_head.chunk_rows * base.config.hidden_size * 2 / 2**20,
            'free_before_mib': free / 2**20,
            'note': 'CUDA context/driver memory is additional to the allocator cap.'}


def enable_available_gpu_dispatch(policy, reserve_gib=1.25):
    """Place layers using current free VRAM; stream CPU-held layers on demand."""
    from accelerate import dispatch_model, infer_auto_device_map
    import os
    if policy.cfg.device != 'cpu':
        raise ValueError('Load the policy on CPU before automatic dispatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one evaluation GPU')
    free, total = torch.cuda.mem_get_info()
    reserve = int(reserve_gib * 2**30)
    budget = free - reserve
    if budget < 1024 * 2**20:
        raise RuntimeError('Less than one GiB remains after generation headroom')
    from tau2_grpo.mlp_chunking import enable_mlp_chunking
    chunked_layers = enable_mlp_chunking(policy.model, 256)
    available_ram = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    mapping = infer_auto_device_map(
        policy.model, max_memory={0: budget, 'cpu': max(16 * 2**30, available_ram // 2)},
        no_split_module_classes=['Cohere2DecoderLayer'], dtype=torch.bfloat16)
    policy.model = dispatch_model(policy.model, device_map=mapping, offload_buffers=True)
    return {'mode': 'automatic_gpu_dispatch_cpu_weight_offload',
            'free_before_gib': free / 2**30, 'generation_reserve_gib': reserve_gib,
            'placement_budget_gib': budget / 2**30,
            'allocator_fraction_cap': None, 'device_map': mapping,
            'inference_mlp_chunk_tokens': 256, 'chunked_layers': chunked_layers,
            'note': 'GPU layers stay resident; CPU-held layers stream to GPU for execution.'}
