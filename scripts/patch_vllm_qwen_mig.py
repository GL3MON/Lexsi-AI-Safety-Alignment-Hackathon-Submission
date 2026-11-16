"""Apply MIG UUID-aware NVML queries to the isolated vLLM 0.17.1 environment."""

import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
site = root / ".venv-vllm-qwen/lib/python3.12/site-packages/vllm/platforms"
for filename in ("interface.py", "cuda.py"):
    p = site / filename
    current = p.read_text()
    if "# Local Qwen MIG UUID support" in current:
        continue
    result = current
    if filename == "interface.py":
        needle = "            return int(physical_device_id)"
        assert result.count(needle) == 1
        result = result.replace(
            needle,
            """            # Local Qwen MIG UUID support
            if cls.device_type == "cuda" and physical_device_id.startswith(("MIG-", "GPU-")):
                return physical_device_id
            return int(physical_device_id)""",
        )
    else:
        result = result.replace("pynvml.nvmlDeviceGetHandleByIndex(", "_nvml_handle(")
        helper = """# Local Qwen MIG UUID support

def _nvml_handle(device_id):
    if isinstance(device_id, str):
        return pynvml.nvmlDeviceGetHandleByUUID(device_id)
    return pynvml.nvmlDeviceGetHandleByIndex(device_id)


"""
        result = result.replace(
            "class NvmlCudaPlatform(CudaPlatformBase):",
            helper + "class NvmlCudaPlatform(CudaPlatformBase):",
            1,
        )
        needle = "major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)"
        assert result.count(needle) == 1
        result = result.replace(
            needle,
            """major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(
                pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
                if isinstance(physical_device_id, str) and physical_device_id.startswith("MIG-")
                else handle)""",
        )
    compile(result, str(p), "exec")
    p.with_suffix(".py.before-mig-patch").write_text(current)
    temporary = p.with_suffix(".py.updated")
    temporary.write_text(result)
    temporary.replace(p)
    (root / "scripts" / f"vllm-0.17.1-mig-{filename}.patch").write_text(
        "".join(
            difflib.unified_diff(
                current.splitlines(True),
                result.splitlines(True),
                fromfile=f"a/vllm/platforms/{filename}",
                tofile=f"b/vllm/platforms/{filename}",
            )
        )
    )
    print("Patched", p)
