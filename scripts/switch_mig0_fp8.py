"""Safely benchmark a retail server's FP8 mode at an episode boundary, with rollback."""

import argparse
import json
import os
import signal
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

from benchmark_qwen_serving import benchmark

ROOT = Path(__file__).resolve().parents[1]
PARALLEL = ROOT / "outputs/retail-data/qwen-sft-parallel"
SOURCE = ROOT / "outputs/retail-data/qwen-sft-fixed-mig0"
BASE_URL = "http://127.0.0.1:8005/v1"
WORKER_INDEX = 1
PORT = 8005
DEVICE = "MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04"
LABEL = "MIG0"
FP8_MEMORY_UTILIZATION = "0.40"
SERVER_LOG = ROOT / "outputs/logs/qwen-retail-server-mig0.log"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def ready():
    try:
        with urllib.request.urlopen(BASE_URL + "/models", timeout=3) as response:
            return any(
                model["id"] == "qwen3.5-4b" for model in json.load(response)["data"]
            )
    except OSError:
        return False


def stop_server(pid):
    if not alive(pid):
        return
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    if (
        "vllm.entrypoints.openai.api_server" not in argv
        or argv[argv.index("--port") + 1] != str(PORT)
        or os.getpgid(pid) != pid
    ):
        raise RuntimeError(f"Refusing to stop an unverified {LABEL} server group")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 45
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if alive(pid):
        os.killpg(pid, signal.SIGKILL)
    time.sleep(3)


def start_server(quantization):
    profile = {
        "QWEN_QUANTIZATION": quantization,
        "QWEN_OFFLOAD": "none" if quantization == "fp8" else "prefetch",
        "QWEN_GPU_MEMORY_UTILIZATION": FP8_MEMORY_UTILIZATION
        if quantization == "fp8"
        else "0.30",
    }
    path = PARALLEL / "server-profiles.json"
    profiles = json.loads(path.read_text()) if path.exists() else {}
    profiles[str(WORKER_INDEX)] = profile
    save(path, profiles)
    env = os.environ.copy()
    env.update(profile)
    env.update(CUDA_VISIBLE_DEVICES=DEVICE, PORT=str(PORT))
    with SERVER_LOG.open("a") as log:
        log.write(
            f"\n{LABEL} serving switch: quantization={quantization}, offload={profile['QWEN_OFFLOAD']}\n"
        )
        log.flush()
        return subprocess.Popen(
            ["bash", "scripts/serve_qwen_retail_mig2.sh"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(server):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if ready():
            return
        if server.poll() is not None:
            raise RuntimeError(
                f"Server exited {server.returncode}; inspect {SERVER_LOG}"
            )
        time.sleep(5)
    raise TimeoutError(f"{LABEL} server did not become ready")


def main():
    global \
        WORKER_INDEX, \
        PORT, \
        DEVICE, \
        LABEL, \
        FP8_MEMORY_UTILIZATION, \
        SERVER_LOG, \
        SOURCE, \
        BASE_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--worker-index", type=int, choices=[0, 1], default=1)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    WORKER_INDEX = args.worker_index
    PORT = [8004, 8005][WORKER_INDEX]
    DEVICE = [
        "MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b",
        "MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04",
    ][WORKER_INDEX]
    LABEL = ["MIG2", "MIG0"][WORKER_INDEX]
    FP8_MEMORY_UTILIZATION = str(
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else [0.33, 0.40][WORKER_INDEX]
    )
    SERVER_LOG = (
        ROOT
        / "outputs/logs"
        / ["qwen-retail-server.log", "qwen-retail-server-mig0.log"][WORKER_INDEX]
    )
    SOURCE = (
        ROOT
        / "outputs/retail-data"
        / ["qwen-sft-fixed", "qwen-sft-fixed-mig0"][WORKER_INDEX]
    )
    BASE_URL = f"http://127.0.0.1:{PORT}/v1"
    output = args.output or ROOT / "outputs/logs" / f"{LABEL.lower()}-fp8-benchmark"
    output.mkdir(parents=True, exist_ok=True)
    marker = PARALLEL / f"worker-{WORKER_INDEX}.maintenance"
    if not marker.exists() or not (SOURCE / "PAUSE_AFTER_EPISODE").exists():
        raise RuntimeError("Schedule a collector boundary pause before this switch")
    print(f"Waiting for the existing {LABEL} episode to finish", flush=True)
    while alive(args.collector_pid):
        time.sleep(5)
    print("Episode saved; measuring idle BF16 baseline", flush=True)
    baseline = benchmark(BASE_URL)
    save(output / "bf16.json", baseline)
    print("BF16 baseline", baseline["passed"], baseline["total_seconds"], flush=True)
    server = None
    report = {
        "accepted_fp8": False,
        "device": LABEL,
        "gpu_memory_utilization": float(FP8_MEMORY_UTILIZATION),
        "baseline": str(output / "bf16.json"),
    }
    resumed = False
    try:
        stop_server(args.server_pid)
        print("Starting full-GPU FP8", flush=True)
        server = start_server("fp8")
        wait_ready(server)
        candidate = benchmark(BASE_URL)
        save(output / "fp8.json", candidate)
        report.update(
            candidate=str(output / "fp8.json"),
            latency_ratio=candidate["total_seconds"] / baseline["total_seconds"],
        )
        if not baseline["passed"] or not candidate["passed"]:
            raise RuntimeError("One or more BF16/FP8 smoke checks failed")
        if candidate["total_seconds"] > baseline["total_seconds"] * 1.10:
            raise RuntimeError("FP8 did not meet the latency regression check")
        report["accepted_fp8"] = True
        print(
            "FP8 accepted",
            {k: v for k, v in candidate.items() if k != "cases"},
            flush=True,
        )
        resumed = True
    except Exception as error:  # noqa: BLE001 -- Any startup/probe failure must roll back serving.
        report["error"] = str(error)
        traceback.print_exc()
        if server is not None:
            stop_server(server.pid)
        print("Rolling back to the original BF16 offloaded server", flush=True)
        server = start_server("none")
        wait_ready(server)
        report["rolled_back"] = True
        resumed = True
    finally:
        report["healthy_endpoint"] = ready()
        save(output / "report.json", report)
        if resumed and report["healthy_endpoint"]:
            (SOURCE / "PAUSE_AFTER_EPISODE").unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            print(f"{LABEL} collection released; supervisor will resume it", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
