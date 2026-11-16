"""Run two disjoint retail collectors and publish their combined selected traces."""

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/retail-data"


class AdoptedCollector:
    """Monitor an existing collector across a supervisor upgrade without stopping it."""

    def __init__(self, pid, source):
        self.pid, self.source = pid, source
        self.identity = self._identity()
        if self.identity is None:
            raise ValueError(f"Collector {pid} is not running")

    def _identity(self):
        try:
            process = Path(f"/proc/{self.pid}")
            if (
                b"scripts/generate_retail_sft.py"
                not in (process / "cmdline").read_bytes()
            ):
                return None
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else fields[19]
        except OSError:
            return None

    def poll(self):
        if self._identity() == self.identity:
            return None
        if (self.source / "PAUSE_AFTER_EPISODE").exists():
            return 0
        path = self.source / "progress.json"
        return (
            0
            if path.exists() and json.loads(path.read_text()).get("generation_complete")
            else 1
        )

    def terminate(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def server_process_present(port):
    """Avoid starting a second server while an existing API process warms up."""
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = process.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if (
            "vllm.entrypoints.openai.api_server" in argv
            and "--port" in argv
            and argv[argv.index("--port") + 1] == str(port)
        ):
            return True
    return False


def merge(sources, output, owners):
    counts = {}
    for partition, filename in [
        ("sft", "sft.jsonl"),
        ("validation", "validation.jsonl"),
    ]:
        combined = []
        seen = set()
        for index, source in enumerate(sources):
            path = source / filename
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["task_id"] not in owners[index]:
                    raise ValueError(f"Unassigned task in {path}: {row['task_id']}")
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                combined.append(row)
        random.Random(2026).shuffle(combined)
        atomic(
            output / filename,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        )
        counts[partition] = dict(Counter(row["task_id"] for row in combined))
    report = {
        "accepted": {key: sum(value.values()) for key, value in counts.items()},
        "per_task": counts,
        "sources": [str(source) for source in sources],
        "policy_review": False,
        "updated_at": time.time(),
    }
    atomic(output / "progress.json", json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-collector-pid", type=int)
    parser.add_argument(
        "--adopt-worker", action="append", default=[], metavar="INDEX:PID"
    )
    parser.add_argument(
        "--manage-servers",
        action="store_true",
        help="Restore missing local vLLM API processes on their assigned MIGs",
    )
    parser.add_argument("--max-worker-restarts", type=int, default=3)
    parser.add_argument("--source", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corrective-prompts", action="store_true")
    parser.add_argument(
        "--collection-mode",
        choices=["successful", "all-generations"],
        default="all-generations",
    )
    args = parser.parse_args()
    sources = args.source or [DATA / "qwen-sft-fixed", DATA / "qwen-sft-fixed-mig0"]
    if len(sources) != 2:
        parser.error("Provide exactly two --source paths")
    sources = [source.resolve() for source in sources]
    output = (args.output or DATA / "qwen-sft-parallel").resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed_path = output / "completed.json"
    if completed_path.exists():
        completed_path.rename(output / f"previous-completed-{time.time_ns()}.json")
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    train = manifest["partitions"]["sft"]
    validation = manifest["partitions"]["validation"]
    # Preserve tasks 2 and 57 on the original collector, including its in-flight episode.
    owners = [
        set(train[:2] + train[2::2] + validation[::2]),
        set(train[3::2] + validation[1::2]),
    ]
    assert not owners[0] & owners[1]
    assert owners[0] | owners[1] == set(train + validation)
    atomic(
        output / "workers.json",
        json.dumps(
            [
                {
                    "device": device,
                    "base_url": url,
                    "output": str(source),
                    "task_ids": sorted(tasks),
                }
                for device, url, source, tasks in zip(
                    ["MIG2", "MIG0"],
                    ["http://127.0.0.1:8004/v1", "http://127.0.0.1:8005/v1"],
                    sources,
                    owners,
                )
            ],
            indent=2,
        ),
    )
    workers = {}
    for assignment in args.adopt_worker:
        index, pid = map(int, assignment.split(":"))
        if index not in (0, 1) or index in workers:
            parser.error("Adopt each worker index at most once")
        workers[index] = AdoptedCollector(pid, sources[index])
        print(f"ADOPT worker={index} pid={pid}", flush=True)
    completed = set()
    failed = {}
    restarts = Counter()
    next_start = {}
    unavailable_since = {}
    server_starts = Counter()
    server_handles = []
    devices = [
        "MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b",
        "MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04",
    ]
    handles = []
    last_counts = None
    try:
        while len(completed) < 2:
            for index, port in enumerate([8004, 8005]):
                if index in workers or index in completed:
                    continue
                if (output / f"worker-{index}.maintenance").exists():
                    continue
                if time.monotonic() < next_start.get(index, 0):
                    continue
                if (
                    index == 0
                    and args.wait_collector_pid
                    and Path(f"/proc/{args.wait_collector_pid}").exists()
                ):
                    continue
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/v1/models", timeout=3
                    ) as response:
                        json.load(response)
                except OSError:
                    unavailable_since.setdefault(index, time.monotonic())
                    if (
                        args.manage_servers
                        and not server_process_present(port)
                        and server_starts[index] < 3
                    ):
                        env = os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=devices[index], PORT=str(port))
                        profiles = output / "server-profiles.json"
                        if profiles.exists():
                            profile = json.loads(profiles.read_text()).get(
                                str(index), {}
                            )
                            env.update({str(k): str(v) for k, v in profile.items()})
                        server_log = (
                            ROOT
                            / f"outputs/logs/qwen-retail-server-recovery-{index}.log"
                        ).open("a")
                        server_handles.append(server_log)
                        server = subprocess.Popen(
                            ["bash", "scripts/serve_qwen_retail_mig2.sh"],
                            cwd=ROOT,
                            env=env,
                            stdout=server_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        server_starts[index] += 1
                        next_start[index] = time.monotonic() + 30
                        print(
                            f"RESTORE server port={port} pid={server.pid}", flush=True
                        )
                    if time.monotonic() - unavailable_since[index] > 1200:
                        failed[index] = (
                            f"Server on port {port} unavailable for 20 minutes"
                        )
                        completed.add(index)
                        print(
                            f"FAILED worker={index}: {failed[index]}; other worker continues",
                            flush=True,
                        )
                    continue
                unavailable_since.pop(index, None)
                (sources[index] / "PAUSE_AFTER_EPISODE").unlink(missing_ok=True)
                command = [
                    "bash",
                    "scripts/generate_corrective_retail_sft.sh" if args.corrective_prompts else "scripts/generate_retail_sft.sh",
                    "--output",
                    str(sources[index]),
                    "--base-url",
                    f"http://127.0.0.1:{port}/v1",
                    "--task-ids",
                    ",".join(sorted(owners[index])),
                    "--collection-mode",
                    args.collection_mode,
                ]
                if sources[index].exists():
                    command.append("--resume")
                log = (ROOT / f"outputs/logs/retail-sft-worker-{index}.log").open("a")
                handles.append(log)
                workers[index] = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                print(
                    f"START worker={index} port={port} pid={workers[index].pid}",
                    flush=True,
                )
            report = merge(sources, output, owners)
            if report["accepted"] != last_counts:
                print("SELECTED", report["accepted"], flush=True)
                last_counts = report["accepted"]
            for index, worker in list(workers.items()):
                status = worker.poll()
                if status is None:
                    continue
                if (output / f"worker-{index}.maintenance").exists():
                    del workers[index]
                    print(
                        f"PAUSED worker={index} for maintenance; other worker continues",
                        flush=True,
                    )
                    continue
                if status:
                    restarts[index] += 1
                    if restarts[index] <= args.max_worker_restarts:
                        next_start[index] = time.monotonic() + 30
                        print(
                            f"RETRY worker={index} exit={status} restart={restarts[index]}; other worker continues",
                            flush=True,
                        )
                    else:
                        failed[index] = (
                            f"Worker exited {status} after {restarts[index]} failures"
                        )
                        completed.add(index)
                        print(
                            f"FAILED worker={index}: {failed[index]}; other worker continues",
                            flush=True,
                        )
                else:
                    completed.add(index)
                del workers[index]
            atomic(
                output / "worker-status.json",
                json.dumps(
                    {
                        "updated_at": time.time(),
                        "running": {i: worker.pid for i, worker in workers.items()},
                        "completed": sorted(completed),
                        "failed": failed,
                        "restarts": dict(restarts),
                        "waiting_for_servers": sorted(unavailable_since),
                    },
                    indent=2,
                ),
            )
            if len(completed) < 2:
                time.sleep(5)
        report = merge(sources, output, owners)
        report["failed_workers"] = failed
        report["generation_complete"] = not failed and all(
            json.loads((source / "progress.json").read_text()).get(
                "generation_complete", False
            )
            for source in sources
        )
        atomic(
            output
            / (
                "completed.json" if report["generation_complete"] else "incomplete.json"
            ),
            json.dumps(report, indent=2),
        )
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                worker.terminate()
        for handle in handles:
            handle.close()
        for handle in server_handles:
            handle.close()
        lock.close()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
