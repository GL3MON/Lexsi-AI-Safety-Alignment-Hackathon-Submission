"""Exercise the real SFT/W&B path on a random tiny CPU model, never base weights."""

import json
import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from tau2_grpo.sft_config import PROJECT, SFTSettings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=["aligntune", "trl", "both"], default="both")
parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
args = parser.parse_args()
root = PROJECT / "outputs" / f"sft-validation-{uuid4().hex[:8]}"
model_dir = root / "fixture_model"
model_dir.mkdir(parents=True)
vocab = {token: index for index, token in enumerate([
    "<pad>", "<unk>", "<s>", "</s>", "system", "user", "assistant",
    "lookup", "order", "name", "arguments", "tool_call", "id", "Find",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "{", "}", ":", '"', ",", "<", ">", "/",
])}
backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
    bos_token="<s>", eos_token="</s>", model_max_length=256,
)
tokenizer.chat_template = (
    "{% for message in messages %}{{ '<s>' + message.role + '\\n' + message.content + '</s>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<s>assistant\\n' }}{% endif %}"
)
tokenizer.save_pretrained(model_dir)
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
))
model.save_pretrained(model_dir)
dataset = root / "fixture.jsonl"
with dataset.open("w") as file:
    for index in range(12):
        file.write(json.dumps({"messages": [
            {"role": "system", "content": "lookup order"},
            {"role": "user", "content": f"Find order {index}"},
            {"role": "assistant", "content": '<tool_call>' + json.dumps({"name": "lookup", "arguments": {"id": index}}) + '</tool_call>'},
        ]}) + "\n")
cfg = asdict(SFTSettings())
cfg.update(
    model_id=str(model_dir), model_revision="main", dataset_path=str(dataset),
    rl_holdout_path=None, output_dir=str(root / "runs"), use_cpu=True, precision="fp32",
    max_length=256, max_steps=2, batch_size=1, gradient_accumulation_steps=1,
    max_train_samples=4, max_eval_samples=2, eval_steps=1, save_steps=1,
    lora_rank=2, lora_alpha=4, lora_target_modules=["q_proj", "v_proj"],
    wandb_mode=args.wandb_mode, wandb_project="tau2-retail", wandb_run_name="aligntune-sft-integration-check",
    log_sample_count=1, sample_max_new_tokens=8, gradient_checkpointing=False,
)
backends = ["trl", "aligntune"] if args.backend == "both" else [args.backend]
runs = {}
for backend_name in backends:
    cfg.update(backend=backend_name, output_dir=str(root / backend_name))
    config_path = root / f"config-{backend_name}.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    subprocess.run([sys.executable, "-m", "tau2_grpo.sft", "--config", str(config_path)], check=True, cwd=PROJECT)
    run = next((root / backend_name).glob("run-*"))
    runs[backend_name] = run
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["changed_adapter_tensors"] > 0
    assert (run / "final_adapter" / "adapter_model.safetensors").is_file()
    assert list((run / "wandb").glob("*/run-*.wandb")), "W&B event log missing"
    history = json.loads((run / "trainer_log_history.json").read_text())
    assert any("loss" in event for event in history)
    assert any("eval_loss" in event for event in history)
    if backend_name == "aligntune":
        events = json.loads((run / "aligntune_events.json").read_text())
        kinds = [e["event"] for e in events]
        assert kinds[0] == "init" and kinds.count("train_begin") == kinds.count("train_end") == 1
        assert kinds.index("train_begin") < kinds.index("train_end")
    print(f"{backend_name} SFT integration passed. Artifacts: {run}")

if args.backend == "both":
    import torch
    from safetensors.torch import load_file

    weights = {name: load_file(str(run / "final_adapter" / "adapter_model.safetensors")) for name, run in runs.items()}
    assert weights["trl"].keys() == weights["aligntune"].keys()
    assert all(torch.equal(weights["trl"][key], weights["aligntune"][key]) for key in weights["trl"])
    losses = {name: json.loads((run / "metrics.json").read_text())["train_loss"] for name, run in runs.items()}
    assert losses["trl"] == losses["aligntune"]
    samples = {name: json.loads((run / "generated_samples.json").read_text()) for name, run in runs.items()}
    assert samples["trl"] == samples["aligntune"]
    (root / "parity.json").write_text(json.dumps({
        "identical_adapter_tensors": True, "identical_training_loss": True,
        "identical_generated_samples": True, "training_loss": losses["trl"],
    }, indent=2))
    print(f"Backend parity passed: byte-identical adapter tensors, equal loss and generated samples. {root / 'parity.json'}")
