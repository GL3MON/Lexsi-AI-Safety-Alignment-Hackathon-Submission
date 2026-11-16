"""Create a hardlinked vLLM 0.8.5 view with Transformers 4 metadata aliases."""

import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--source",
    type=Path,
    default=Path("outputs/hf-export/tiny-aya-global-function-calling-sft-525"),
)
p.add_argument("--output", type=Path, default=Path("outputs/vllm-model-525"))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for source in a.source.iterdir():
    if (
        source.is_file()
        and source.name not in ("config.json", "tokenizer_config.json")
        and not (a.output / source.name).exists()
    ):
        os.link(source, a.output / source.name)
config = json.loads((a.source / "config.json").read_text())
assert (
    config["model_type"] == "cohere2"
    and config["use_gated_activation"]
    and config["use_parallel_block"]
)
config["rope_theta"] = config["rope_parameters"]["rope_theta"]
config["interleaved_sliding_window"] = config["sliding_window"]
config["torch_dtype"] = config["dtype"]
(a.output / "config.json").write_text(json.dumps(config, indent=2))
config = json.loads((a.source / "tokenizer_config.json").read_text())
config["tokenizer_class"] = "CohereTokenizerFast"
(a.output / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
print(a.output)
