"""Make vLLM's existing Triton GDN prefill selectable when nvcc is unavailable."""

import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = (
    root
    / ".venv-vllm-qwen/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py"
)
s = p.read_text()
if "# Local optional Triton GDN fallback" not in s:
    original = s
    s = s.replace("import torch\n", "import os\n\nimport torch\n", 1)
    needle = "        if current_platform.is_cuda() and current_platform.is_device_capability(90):"
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        """        # Local optional Triton GDN fallback: use the bundled FLA implementation.
        if (current_platform.is_cuda() and current_platform.is_device_capability(90)
                and os.environ.get("TAU2_QWEN_GDN_TRITON") != "1"):""",
    )
    compile(s, str(p), "exec")
    p.with_suffix(".py.before-gdn-patch").write_text(original)
    tmp = p.with_suffix(".py.updated")
    tmp.write_text(s)
    tmp.replace(p)
    (root / "scripts/vllm-0.17.1-qwen-gdn.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                s.splitlines(True),
                fromfile="a/vllm/model_executor/models/qwen3_next.py",
                tofile="b/vllm/model_executor/models/qwen3_next.py",
            )
        )
    )
    print("Patched optional Triton GDN prefill")
