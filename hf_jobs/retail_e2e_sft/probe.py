"""Find a measured micro-batch boundary with isolated real optimizer steps."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from tau2_grpo.sft_config import load_settings, resolve_path, settings_dict


def worker(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer
    from tau2_grpo.sft_loss import configure_chunked_loss

    cfg = load_settings(args.config)
    result = {"batch_size": args.batch_size}
    try:
        set_seed(cfg.seed)
        free, total = torch.cuda.mem_get_info()
        result.update(free_before_model_gib=free / 2**30, total_gpu_gib=total / 2**30)
        specimen = json.loads(Path(args.sample_file).read_text())
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        configure_chunked_loss(cfg.loss_chunk_size, cfg.loss_projection_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.model_revision, dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation,
        )
        model.config.use_cache = False
        if cfg.initial_adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,str(resolve_path(cfg.initial_adapter)),is_trainable=True)
        if cfg.mlp_chunk_size:
            from tau2_grpo.mlp_chunking import enable_mlp_chunking
            enable_mlp_chunking(model,cfg.mlp_chunk_size)
        training = SFTConfig(
            output_dir=str(Path(args.result_path).with_suffix("")),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1, max_steps=2, learning_rate=cfg.learning_rate,
            warmup_steps=0, bf16=True,
            gradient_checkpointing=cfg.gradient_checkpointing,
            activation_offloading=cfg.activation_offloading,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=cfg.max_length, loss_type=cfg.loss_type,
            dataset_kwargs={"skip_prepare_dataset": True},
            completion_only_loss=False, assistant_only_loss=False,
            report_to=[], save_strategy="no", eval_strategy="no",
            logging_steps=1, disable_tqdm=True, optim=cfg.optim,
            dataloader_num_workers=0,remove_unused_columns=not cfg.example_normalized_loss,
        )
        trainer_type=SFTTrainer
        if cfg.example_normalized_loss:
            from tau2_grpo.sft_weighted import trainer_class
            trainer_type=trainer_class()
        trainer = trainer_type(
            model=model, args=training, processing_class=tokenizer,
            train_dataset=Dataset.from_list([specimen] * (2 * args.batch_size)),
            peft_config=None if cfg.initial_adapter else LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules,
                task_type="CAUSAL_LM", bias="none",
            ),
        )
        trainer.weighted_chunk_size=cfg.loss_chunk_size
        torch.cuda.reset_peak_memory_stats()
        output = trainer.train()
        torch.cuda.synchronize()
        result.update(
            status="ok", train_loss=output.training_loss,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            step_time_seconds=output.metrics["train_runtime"] / 2,
        )
    except Exception as error:
        import traceback
        traceback.print_exc()
        text = str(error)
        # This host hides NVML process information. PyTorch's OOM diagnostics can
        # themselves fail with an NVML allocator assertion instead of OutOfMemoryError.
        oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in text.lower()
        nvml = "NVML_SUCCESS" in text and "CUDACachingAllocator" in text
        result.update(status="oom" if oom else "allocator_error" if nvml else "error", error=text)
        if torch.cuda.is_initialized():
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    Path(args.result_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.json")
    parser.add_argument("--sample-manifest")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--label", default="longest")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--sample-file")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--result-path")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.max_batch_size < 1:
        parser.error("max-batch-size must be positive")
    import wandb
    from transformers import AutoTokenizer
    from tau2_grpo.sft_data import read_records, tokenize_assistant_only

    cfg = load_settings(args.config)
    if cfg.precision != "bf16" or cfg.use_cpu:
        raise ValueError("This probe currently tests the bf16 CUDA SFT configuration")
    root = resolve_path("outputs/sft-diagnostics") / f"batch-{args.label}-{uuid4().hex[:8]}"
    root.mkdir(parents=True)
    records = read_records(resolve_path(cfg.dataset_path))
    if args.sample_manifest:
        ids = {row["id"] for row in json.loads(Path(args.sample_manifest).read_text())["train"]}
        records = [r for r in records if r["id"] in ids]
    if not records:
        raise ValueError("No probe records")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    if cfg.chat_template_path:
        tokenizer.chat_template=resolve_path(cfg.chat_template_path).read_text()
    # Source token_count metadata was generated with this same Tiny Aya tokenizer.
    # Re-encode the selected record to verify the actual shape used by the worker.
    selected = max(records, key=lambda r: r.get("context_tokens",r.get("token_count",0)))
    specimen, _ = tokenize_assistant_only(tokenizer, selected["messages"], cfg.max_length)
    sample_file = root / "sample.json"
    sample_file.write_text(json.dumps(specimen))
    metadata = {
        "source_id": selected["id"], "tokens_per_sample": len(specimen["input_ids"]),
        "supervised_tokens_per_sample": sum(x != -100 for x in specimen["labels"]),
        "label": args.label, "config": args.config,
        "settings": settings_dict(cfg),
        "scope": "two full optimizer steps, repeated longest selected conversation, actual LoRA SFT settings",
    }
    (root / "workload.json").write_text(json.dumps(metadata, indent=2))
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    run = wandb.init(
        project=cfg.wandb_project, entity=cfg.wandb_entity,
        name=f"tiny-aya-batch-probe-{args.label}", mode=cfg.wandb_mode,
        job_type="batch-probe", config=metadata, dir=str(root),
    )
    print(f"Probe artifacts: {root}\nW&B: {run.url}", flush=True)
    results = []

    def attempt(batch):
        path = root / f"batch-{batch}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--config", str(resolve_path(args.config)), "--sample-file", str(sample_file),
                   "--batch-size", str(batch), "--result-path", str(path)]
        print(f"Testing batch {batch}, {metadata['tokens_per_sample']} tokens/sample", flush=True)
        with (root / f"batch-{batch}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(path.read_text())
        results.append(result)
        print(json.dumps(result), flush=True)
        run.log({"probe/batch_size": batch, "probe/fit": int(result["status"] == "ok"),
                 **{f"probe/{key}": result[key] for key in ("peak_allocated_gib", "peak_reserved_gib", "step_time_seconds") if key in result}})
        if result["status"] == "error":
            raise RuntimeError(f"Probe error (not an OOM): {result['error']}")
        return result["status"] == "ok"

    success = False
    try:
        low, high = 0, None
        batch = 1
        while batch <= args.max_batch_size:
            if not attempt(batch):
                high = batch
                break
            low = batch
            if batch == args.max_batch_size:
                break
            batch = min(batch * 2, args.max_batch_size)
        if high is not None:
            while high - low > 1:
                middle = (low + high) // 2
                if attempt(middle):
                    low = middle
                else:
                    high = middle
        summary = {**metadata, "largest_successful_batch": low, "smallest_failed_batch": high,
                   "boundary_found": high is not None, "attempts": results, "wandb_url": run.url}
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        run.summary.update({"largest_successful_batch": low, "smallest_failed_batch": high or 0})
        run.log({"probe/results": wandb.Table(
            columns=["batch_size", "status", "peak_allocated_gib", "peak_reserved_gib", "error"],
            data=[[r["batch_size"], r["status"], r.get("peak_allocated_gib"), r.get("peak_reserved_gib"), r.get("error")] for r in results],
        )})
        print(f"Batch summary: {root / 'summary.json'}", flush=True)
        success = True
    finally:
        run.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
