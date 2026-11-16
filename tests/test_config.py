import ast
import json
import tempfile
import unittest
from pathlib import Path

from tau2_grpo.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def load(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw))
            return load_config(path)

    def test_presets(self):
        for path in (ROOT / "configs").glob("*.json"):
            if path.name.startswith("sft"):
                from tau2_grpo.sft_config import load_settings

                self.assertEqual(load_settings(path).wandb_mode, "online")
                continue
            if path.name.startswith("rl-retail"):
                from tau2_grpo.rl_config import load_settings

                self.assertGreaterEqual(load_settings(path).group_size, 2)
                continue
            if path.name == "rl-glaive.json":
                from tau2_grpo.rl_config import load_settings
                raw=json.loads(path.read_text());raw.pop("dataset_path")
                with tempfile.NamedTemporaryFile(mode="w",suffix=".json") as f:
                    json.dump(raw,f);f.flush()
                    self.assertGreaterEqual(load_settings(f.name).group_size,2)
                continue
            if path.name == "retail-generation.json":
                # Serving/collection provenance, not an ExperimentConfig preset.
                continue
            self.assertGreaterEqual(load_config(path).group_size, 2)

    def test_invalid_settings(self):
        for raw in [
            {"group_size": 1},
            {"train_steps": 0},
            {"domain": "unsupported"},
            {"group_size": True},
            {"learning_rate": -1},
            {"typo": 1},
            {"max_completion_tokens": 40000},
            {"max_tool_iterations": 1},
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load(raw)

    def test_workflow_syntax(self):
        for path in (ROOT / "src").rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))

    def test_retail_preview_config(self):
        self.assertEqual(self.load({"domain": "retail"}).domain, "retail")


if __name__ == "__main__":
    unittest.main()
