"""Use native fused LayerNorm without Triton compilation in vLLM 0.8.5."""

import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = (
    root
    / ".venv-vllm/lib/python3.12/site-packages/vllm/model_executor/models/commandr.py"
)
s = p.read_text()
start = (
    s.index("@torch.compile(backend=current_platform.simple_compile_backend)")
    if "@torch.compile(backend=current_platform.simple_compile_backend)" in s
    else -1
)
if start >= 0:
    end = s.index("\n\nclass LayerNorm", start)
    new = """# Local native LayerNorm: BF16 inputs use FP32 accumulation in PyTorch's CUDA kernel.
def layer_norm_func(hidden_states, weight, variance_epsilon):
    return torch.nn.functional.layer_norm(
        hidden_states, (hidden_states.shape[-1],), weight, None, variance_epsilon)
"""
    result = s[:start] + new + s[end:]
    p.with_suffix(".py.before-norm-patch").write_text(s)
    compile(result, str(p), "exec")
    p.write_text(result)
    (root / "scripts/vllm-0.8.5-cohere-norm.patch").write_text(
        "".join(
            difflib.unified_diff(
                s.splitlines(True),
                result.splitlines(True),
                fromfile="a/vllm/model_executor/models/commandr.py",
                tofile="b/vllm/model_executor/models/commandr.py",
            )
        )
    )
    print("Patched native LayerNorm")
else:
    assert "# Local native LayerNorm" in s
    print("LayerNorm patch already applied")
