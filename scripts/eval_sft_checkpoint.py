"""Check held-out tool-call generation from a real saved SFT checkpoint."""

import argparse
import contextlib
import hashlib
import json
import random
import shutil
from pathlib import Path
from uuid import uuid4

from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.sft_config import PROJECT, resolve_path
from tau2_grpo.sft_data import read_records


def latest_checkpoint(root):
    complete = []
    for path in root.glob("run-*/checkpoint-*"):
        try:
            state = json.loads((path / "trainer_state.json").read_text())
            weight = path / "adapter_model.safetensors"
            if weight.is_file() and state["global_step"] == int(path.name.split("-")[-1]):
                complete.append(path)
        except (FileNotFoundError, ValueError, KeyError):
            continue
    if not complete:
        raise ValueError(f"No complete SFT checkpoint under {root}")
    return max(complete, key=lambda path: (path / "adapter_model.safetensors").stat().st_mtime_ns)


def score(text, expected, tools):
    from jsonschema import Draft7Validator

    try:
        actual = parse_calls(text)
        parse_error = None
    except ValueError as error:
        actual, parse_error = [], str(error)
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    schema_errors = []
    for call in actual:
        if call["name"] not in functions:
            schema_errors.append(f"Unknown tool: {call['name']}")
            continue
        schema_errors.extend(str(error.message) for error in Draft7Validator(
            functions[call["name"]]["parameters"]
        ).iter_errors(call["arguments"]))
    syntax_valid = parse_error is None and (bool(actual) if expected else bool(text.strip()))
    return {
        "actual_calls": actual, "parse_error": parse_error,
        "syntax_valid": syntax_valid,
        "schema_valid": syntax_valid and not schema_errors,
        "schema_errors": schema_errors,
        "exact_calls": syntax_valid and actual == expected,
    }


def generation_stop_ids(native_eos, tokenizer):
    stop_ids = set(native_eos if isinstance(native_eos, list) else [native_eos])
    stop_ids.add(tokenizer.eos_token_id)
    end_response = tokenizer.convert_tokens_to_ids("<|END_RESPONSE|>")
    if end_response is not None and end_response != tokenizer.unk_token_id:
        stop_ids.add(end_response)
    stop_ids.discard(None)
    return sorted(stop_ids)


def main():
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--run-root", type=Path, default=PROJECT / "outputs/sft")
    parser.add_argument("--single", type=int, default=6)
    parser.add_argument("--multi", type=int, default=6)
    parser.add_argument("--followup", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    args = parser.parse_args()
    if min(args.single, args.multi, args.followup) < 0 or args.max_new_tokens < 64:
        parser.error("Counts must be nonnegative; max-new-tokens must be at least 64")
    checkpoint = latest_checkpoint(args.run_root) if args.checkpoint == "latest" else resolve_path(args.checkpoint)
    source_run = checkpoint.parent
    if not (source_run / "config.json").exists() and (source_run / "settings.json").exists():
        # Re-evaluate an immutable snapshot after training rotates the original.
        prior = json.loads((source_run / "settings.json").read_text())
        source_run = Path(prior.get("source_run", str(Path(prior["checkpoint"]).parent)))
    cfg = json.loads((source_run / "config.json").read_text())
    runtime = json.loads((source_run / "runtime.json").read_text())
    manifest = json.loads((source_run / "data_manifest.json").read_text())
    out = PROJECT / "outputs/checkpoint-eval" / f"run-{uuid4().hex[:10]}"
    snapshot = out / "checkpoint_snapshot"
    snapshot.mkdir(parents=True)
    for name in (
        "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
        "trainer_state.json", "adapter_model.safetensors",
    ):
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, snapshot / name)
    print(f"Checkpoint: {checkpoint}\nEvaluation outputs: {out}", flush=True)
    state = json.loads((snapshot / "trainer_state.json").read_text())
    data_path = resolve_path(cfg["dataset_path"])
    source_data_hash = json.loads((source_run / "data_summary.json").read_text())["source_sha256"]
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != source_data_hash:
        raise ValueError("SFT source data changed since training")
    validation_ids = {r["id"] for r in manifest["eval"]}
    training_groups = {r["group_id"] for r in manifest["train"]}
    records = [r for r in read_records(data_path) if r["id"] in validation_ids]
    if not records or any(r["group_id"] in training_groups for r in records):
        raise ValueError("Missing validation records or overlap with training groups")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    pools = {"single_turn_call": [], "multi_turn_call": [], "multi_turn_followup": []}
    for record in records:
        indices = [i for i, message in enumerate(record["messages"]) if message["role"] == "assistant"]
        for position, index in enumerate(indices):
            target = record["messages"][index]["content"]
            expected = parse_calls(target)
            if len(tokenizer.encode(target, add_special_tokens=False)) > args.max_new_tokens - 32:
                continue
            if len(indices) == 1 and expected:
                kind = "single_turn_call"
            elif position == 0 and expected:
                kind = "multi_turn_call"
            elif position > 0 and not expected:
                kind = "multi_turn_followup"
            else:
                continue
            pools[kind].append({
                "id": record["id"], "group_id": record["group_id"], "kind": kind,
                "assistant_index": index, "prompt": record["messages"][:index],
                "reference": target, "expected_calls": expected, "tools": record["tools"],
            })
    rng = random.Random(args.seed)
    cases = []
    for kind, count in zip(pools, (args.single, args.multi, args.followup), strict=True):
        rng.shuffle(pools[kind])
        if len(pools[kind]) < count:
            raise ValueError(f"Only {len(pools[kind])} eligible {kind} cases, requested {count}")
        cases.extend(pools[kind][:count])
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    settings = {
        "checkpoint": str(checkpoint), "step": state["global_step"],
        "source_run": str(source_run),
        "model_id": cfg["model_id"], "revision": runtime["resolved_model_revision"],
        "adapter_sha256": hashlib.sha256((snapshot / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "validation_conversations": len(records), "cases": len(cases),
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "scope": "SFT validation; offline next-turn generation; recorded tool results for follow-up prompts",
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    tracking = wandb.init(
        project=cfg["wandb_project"], entity=cfg.get("wandb_entity"),
        name=f"tiny-aya-checkpoint-{state['global_step']}-tool-eval",
        job_type="checkpoint-eval", tags=["function-calling", "heldout", "eval"],
        mode=args.wandb_mode, config=settings, dir=str(out),
    )
    success = False
    try:
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], revision=runtime["resolved_model_revision"],
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
        model = PeftModel.from_pretrained(base, snapshot, is_trainable=False).eval()
        # Tiny Aya's base model stops at END_RESPONSE, while the SFT template
        # also teaches END_OF_TURN. Respect both in baseline and adapter runs.
        stop_ids = generation_stop_ids(model.generation_config.eos_token_id, tokenizer)
        model.generation_config.eos_token_id = stop_ids
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        rows = []
        for number, case in enumerate(cases, 1):
            result = {**case, "outputs": {}}
            for mode in ("base", "sft"):
                inputs = tokenizer.apply_chat_template(
                    case["prompt"], tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to("cuda")
                prompt_tokens = inputs["input_ids"].shape[1]
                if prompt_tokens + args.max_new_tokens > base.config.max_position_embeddings:
                    raise ValueError("Evaluation prompt exceeds context budget")
                adapter_context = model.disable_adapter() if mode == "base" else contextlib.nullcontext()
                with adapter_context, torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        use_cache=True, logits_to_keep=1,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
                    )[0, prompt_tokens:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                result["outputs"][mode] = {
                    "text": text, "raw_text": tokenizer.decode(generated, skip_special_tokens=False),
                    "prompt_tokens": prompt_tokens, "generated_tokens": len(generated),
                    "hit_token_limit": len(generated) == args.max_new_tokens and generated[-1].item() not in stop_ids,
                    **score(text, case["expected_calls"], case["tools"]),
                }
                print(f"{number}/{len(cases)} {case['kind']} {mode}: format={result['outputs'][mode]['syntax_valid']} exact_calls={result['outputs'][mode]['exact_calls']}", flush=True)
            rows.append(result)
            with (out / "results.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")
            with (out / "generations.txt").open("a") as file:
                file.write(f"CASE {number}: {case['kind']}\nREFERENCE:\n{case['reference']}\n\nBASE:\n{result['outputs']['base']['text']}\n\nSFT:\n{result['outputs']['sft']['text']}\n\n{'=' * 72}\n\n")
        summary = {"settings": settings, "wandb_url": tracking.url, "metrics": {}}
        for mode in ("base", "sft"):
            calls = [r["outputs"][mode] for r in rows if r["expected_calls"]]
            followups = [r["outputs"][mode] for r in rows if not r["expected_calls"]]
            summary["metrics"][mode] = {
                "call_cases": len(calls), "valid_call_format": sum(r["syntax_valid"] for r in calls),
                "schema_valid_calls": sum(r["schema_valid"] for r in calls),
                "exact_tool_names_and_arguments": sum(r["exact_calls"] for r in calls),
                "followup_cases": len(followups), "correct_no_call_followups": sum(r["exact_calls"] for r in followups),
                "token_limit_hits": sum(r["outputs"][mode]["hit_token_limit"] for r in rows),
            }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        tracking.summary.update({f"{mode}/{key}": value for mode, values in summary["metrics"].items() for key, value in values.items()})
        tracking.log({"tool_call_outputs": wandb.Table(
            columns=["kind", "id", "reference", "base_output", "sft_output", "base_exact_calls", "sft_exact_calls"],
            data=[[r["kind"], r["id"], r["reference"], r["outputs"]["base"]["text"], r["outputs"]["sft"]["text"], r["outputs"]["base"]["exact_calls"], r["outputs"]["sft"]["exact_calls"]] for r in rows],
        )})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Readable model outputs: {out / 'generations.txt'}", flush=True)
        success = True
    finally:
        tracking.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
