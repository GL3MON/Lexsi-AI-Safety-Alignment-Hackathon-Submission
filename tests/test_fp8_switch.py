import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FP8SwitchTests(unittest.TestCase):
    def test_mig2_profile_keeps_mig0_settings_and_releases_only_mig2(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with patch.object(sys, "path", [str(scripts)] + sys.path):
            spec = importlib.util.spec_from_file_location(
                "fp8_switch", scripts / "switch_mig0_fp8.py"
            )
            switch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(switch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parallel = root / "outputs/retail-data/qwen-sft-parallel"
            source = root / "outputs/retail-data/qwen-sft-fixed"
            parallel.mkdir(parents=True)
            source.mkdir()
            (root / "outputs/logs").mkdir()
            (parallel / "worker-0.maintenance").touch()
            (source / "PAUSE_AFTER_EPISODE").touch()
            original = {
                "QWEN_QUANTIZATION": "fp8",
                "QWEN_OFFLOAD": "none",
                "QWEN_GPU_MEMORY_UTILIZATION": "0.40",
            }
            (parallel / "server-profiles.json").write_text(json.dumps({"1": original}))
            with (
                patch.object(switch, "ROOT", root),
                patch.object(switch, "PARALLEL", parallel),
                patch.object(
                    sys,
                    "argv",
                    [
                        "switch",
                        "--worker-index",
                        "0",
                        "--collector-pid",
                        "123",
                        "--server-pid",
                        "456",
                    ],
                ),
                patch.object(switch, "alive", return_value=False),
                patch.object(switch, "ready", return_value=True),
                patch.object(switch, "stop_server") as stop,
                patch.object(switch, "wait_ready"),
                patch.object(
                    switch.subprocess, "Popen", return_value=SimpleNamespace(pid=888)
                ) as launch,
                patch.object(
                    switch,
                    "benchmark",
                    side_effect=[
                        {"passed": True, "total_seconds": 10},
                        {"passed": True, "total_seconds": 2},
                    ],
                ),
            ):
                switch.main()
            profiles = json.loads((parallel / "server-profiles.json").read_text())
            self.assertEqual(profiles["1"], original)
            self.assertEqual(profiles["0"]["QWEN_OFFLOAD"], "none")
            self.assertEqual(float(profiles["0"]["QWEN_GPU_MEMORY_UTILIZATION"]), 0.33)
            self.assertEqual(launch.call_args.kwargs["env"]["PORT"], "8004")
            self.assertEqual(
                launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"],
                "MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b",
            )
            stop.assert_called_once_with(456)
            self.assertFalse((parallel / "worker-0.maintenance").exists())
            report = json.loads(
                (root / "outputs/logs/mig2-fp8-benchmark/report.json").read_text()
            )
            self.assertTrue(report["accepted_fp8"])
