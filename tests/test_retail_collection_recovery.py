import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class CollectionRecoveryTests(unittest.TestCase):
    def test_adopted_worker_identity_and_planned_pause(self):
        supervisor = module("run_parallel_retail_sft")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with patch.object(
                supervisor.AdoptedCollector, "_identity", return_value="original"
            ):
                worker = supervisor.AdoptedCollector(123, source)
                self.assertIsNone(worker.poll())
            with patch.object(worker, "_identity", return_value=None):
                self.assertEqual(worker.poll(), 1)
                (source / "PAUSE_AFTER_EPISODE").touch()
                self.assertEqual(worker.poll(), 0)
                (source / "PAUSE_AFTER_EPISODE").unlink()
                (source / "progress.json").write_text('{"generation_complete":true}')
                self.assertEqual(worker.poll(), 0)

    def test_generation_counts_include_reward_zero_but_exclude_connection_failures(
        self,
    ):
        collector = module("generate_retail_sft")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "settings.json").write_text(json.dumps({"split": "sft"}))
            records = [
                {"task_id": "16", "reward": 0.0},
                {"task_id": "16", "reward": 1.0},
                {"task_id": "16", "reward": None, "error": "connection refused"},
            ]
            (run / "rewards.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
            self.assertEqual(collector.generation_counts([str(run)]), {"sft:16": 2})

    def test_failed_worker_restarts_without_terminating_healthy_worker(self):
        supervisor = module("run_parallel_retail_sft")
        created = []
        starts = {8004: 0, 8005: 0}

        class Worker:
            def __init__(self, port, status):
                self.port, self.status = port, status
                self.pid, self.polls, self.terminated = len(created) + 100, 0, False

            def poll(self):
                self.polls += 1
                if self.port == 8005 and self.polls == 1:
                    return None
                return self.status

            def terminate(self):
                self.terminated = True

        def start(command, **kwargs):
            port = int(
                command[command.index("--base-url") + 1].split(":")[-1].split("/")[0]
            )
            starts[port] += 1
            worker = Worker(port, 1 if port == 8004 and starts[port] == 1 else 0)
            created.append(worker)
            return worker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "outputs/retail-data"
            data.mkdir(parents=True)
            (root / "outputs/logs").mkdir()
            (data / "split_manifest.json").write_text(
                json.dumps(
                    {"partitions": {"sft": ["2", "57"], "validation": ["69", "34"]}}
                )
            )
            for name in ("qwen-sft-fixed", "qwen-sft-fixed-mig0"):
                path = data / name
                path.mkdir()
                (path / "progress.json").write_text('{"generation_complete":true}')
            with (
                patch.object(supervisor, "ROOT", root),
                patch.object(supervisor, "DATA", data),
                patch("sys.argv", ["supervisor", "--max-worker-restarts", "1"]),
                patch.object(supervisor.subprocess, "Popen", side_effect=start),
                patch.object(
                    supervisor.urllib.request,
                    "urlopen",
                    side_effect=lambda *a, **k: io.StringIO("{}"),
                ),
                patch.object(supervisor.time, "sleep"),
                patch.object(
                    supervisor.time, "monotonic", side_effect=iter(range(0, 10000, 31))
                ),
            ):
                supervisor.main()
            self.assertEqual(starts, {8004: 2, 8005: 1})
            self.assertFalse(any(worker.terminated for worker in created))
            self.assertTrue((data / "qwen-sft-parallel/completed.json").exists())
