"""Patch vLLM 0.8.5 NVML lookup to accept CUDA-visible GPU and MIG UUIDs."""

import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = root / ".venv-vllm/lib/python3.12/site-packages/vllm/platforms/cuda.py"
s = (
    p.with_suffix(".py.before-mig-patch").read_text()
    if p.with_suffix(".py.before-mig-patch").exists()
    else p.read_text()
)
if "# Local MIG UUID support" in s:
    print("MIG UUID patch already applied")
else:
    original = s
    s = s.replace(
        "def device_id_to_physical_device_id(device_id: int) -> int:",
        "def device_id_to_physical_device_id(device_id: int) -> Union[int, str]:",
    )
    needle = "        return int(physical_device_id)"
    assert s.count(needle) == 1
    s = s.replace(
        needle,
        """        # Local MIG UUID support: preserve CUDA's explicitly selected device.
        if physical_device_id.startswith(("MIG-", "GPU-")):
            return physical_device_id
        return int(physical_device_id)""",
    )
    s = s.replace("pynvml.nvmlDeviceGetHandleByIndex(", "_nvml_handle(")
    helper = '''def _nvml_handle(device_id):
    """Resolve a physical ordinal or GPU/MIG UUID without remapping partitions."""
    if isinstance(device_id, str):
        return pynvml.nvmlDeviceGetHandleByUUID(device_id)
    return pynvml.nvmlDeviceGetHandleByIndex(device_id)


'''
    s = s.replace(
        "def device_id_to_physical_device_id(",
        helper + "def device_id_to_physical_device_id(",
        1,
    )
    s = s.replace(
        "major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)",
        'major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(\n                pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)\n                if isinstance(physical_device_id, str) and physical_device_id.startswith("MIG-") else handle)',
    )
    backup = p.with_suffix(".py.before-mig-patch")
    if not backup.exists():
        backup.write_text(original)
    compile(s, str(p), "exec")
    p.write_text(s)
    patch = root / "scripts/vllm-0.8.5-mig-uuid.patch"
    patch.write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                s.splitlines(True),
                fromfile="a/vllm/platforms/cuda.py",
                tofile="b/vllm/platforms/cuda.py",
            )
        )
    )
    print("Patched", p)
