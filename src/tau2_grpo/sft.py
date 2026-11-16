"""Configurable LoRA SFT with verified assistant labels and mandatory W&B logging."""

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from importlib.metadata import version
from uuid import uuid4

from tau2_grpo.prepare_function_calling import digest
from tau2_grpo.sft_config import PROJECT, load_settings, resolve_path, settings_dict
from tau2_grpo.sft_data import read_records, split_validation, tokenize_assistant_only, validate_task_partitions, encode_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/sft.json"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--wandb-mode", choices=["online", "offline"])
    parser.add_argument("--backend", choices=["aligntune", "trl", "agenttune"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", metavar="CHECKPOINT")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        try:
            key, value = item.split("=", 1)
            overrides[key] = json.loads(value)
        except ValueError:
            parser.error(f"Expected --set KEY=JSON; received {item!r}")
    for name, value in (
        ("wandb_mode", args.wandb_mode), ("max_steps", args.max_steps),
        ("resume_from_checkpoint", args.resume),
        ("backend", args.backend),
    ):
        if value is not None:
            overrides[name] = value
    cfg = load_settings(args.config, overrides)
    if args.print_config:
        print(json.dumps(settings_dict(cfg), indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This SFT entry point supports one process; do not use a distributed launcher")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This entry point deliberately does not inherit GRPO's disabled W&B mode.
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        raise ValueError("Unset WANDB_DISABLED: this SFT trainer requires W&B logging")
    from transformers import AutoConfig, AutoTokenizer

    resume = resolve_path(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else None
    if resume and (not resume.is_dir() or not (resume / "trainer_state.json").is_file()):
        raise ValueError(f"Not a Trainer checkpoint: {resume}")
    if resume and args.prepare_only:
        parser.error("Use prepare-only without resume")
    run_dir = resume.parent if resume else resolve_path(cfg.output_dir) / f"run-{uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=bool(resume))
    print(f"SFT artifacts: {run_dir}", flush=True)
    dataset_path = resolve_path(cfg.dataset_path)
    records = read_records(dataset_path)
    eval_path = resolve_path(cfg.eval_dataset_path) if cfg.eval_dataset_path else None
    task_manifest_path = resolve_path(cfg.task_split_manifest) if cfg.task_split_manifest else None
    if eval_path:
        train_records, eval_records = records, read_records(eval_path)
    else:
        train_records, eval_records = split_validation(records, cfg.validation_fraction, cfg.seed)
    if task_manifest_path:
        validate_task_partitions(train_records, eval_records, json.loads(task_manifest_path.read_text()))
    holdout_path = resolve_path(cfg.rl_holdout_path) if cfg.rl_holdout_path else None
    # if holdout_path:
    #     reserved = read_records(holdout_path)
    #     if {r["group_id"] for r in train_records + eval_records} & {r["group_id"] for r in reserved}:
    #         raise ValueError("SFT data overlaps the reserved RL prompt groups")
    if cfg.train_order == "tool_call_count":
        from tau2_grpo.sft_data import order_by_tool_calls
        train_records = order_by_tool_calls(train_records)
    if cfg.max_train_samples:
        train_records = train_records[:cfg.max_train_samples]
    if cfg.max_eval_samples:
        eval_records = eval_records[:cfg.max_eval_samples]
    if cfg.overfit_debug:
        # Explicit diagnostic only: measure memorization on the same examples.
        eval_records = list(train_records)
        print("OVERFIT DEBUG: evaluation intentionally uses the training examples.", flush=True)
    base_config = AutoConfig.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    revision = getattr(base_config, "_commit_hash", None) or cfg.model_revision
    extension=None
    if cfg.rope_type:
        from tau2_grpo.context_extension import extend_config,save_extension
        base_config,extension=extend_config(base_config,cfg.max_length,cfg.rope_type,cfg.rope_factor,
            cfg.rope_original_max_position_embeddings,cfg.rope_attention_factor)
        save_extension(run_dir/'model-config',base_config,extension)
    if cfg.max_length > base_config.max_position_embeddings:
        raise ValueError("max_length exceeds the base model context limit")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=revision)
    if extension:tokenizer.model_max_length=cfg.max_length
    declared_templates = {r["chat_template_path"] for r in train_records + eval_records if r.get("chat_template_path")}
    if not cfg.chat_template_path and len(declared_templates) > 1:
        raise ValueError("Dataset declares conflicting chat templates; set chat_template_path explicitly")
    template_path = cfg.chat_template_path or next(iter(declared_templates), None)
    if template_path:
        tokenizer.chat_template = resolve_path(template_path).read_text()
    (run_dir / "chat_template.jinja").write_text(tokenizer.chat_template)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenized, used, dropped, mask_previews = {}, {}, [], []
    for split, pool in (("train", train_records), ("eval", eval_records)):
        tokenized[split], used[split] = [], []
        for index, (record, encoded, spans, error) in enumerate(encode_records(tokenizer,pool,cfg.max_length,cfg.preprocessing_num_workers)):
            if error:
                if cfg.overlength_policy == "drop" and error.startswith("overlength:"):
                    dropped.append({"id": record["id"], "split": split, "reason": error})
                    continue
                raise ValueError(f"{split} conversation {record['id']}: {error}")
            if cfg.example_normalized_loss:
                weight=record.get('loss_weight')
                if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:
                    raise ValueError('Weighted SFT requires a positive loss_weight on every record')
                encoded['loss_weight']=float(weight)
            tokenized[split].append(encoded)
            used[split].append(record)
            if len(mask_previews) < 2:
                mask_previews.append({
                    "id": record["id"], "split": split, "assistant_spans": spans,
                    "total_tokens": len(encoded["input_ids"]),
                    "supervised_tokens": sum(v != -100 for v in encoded["labels"]),
                    "assistant_targets": [tokenizer.decode(encoded["input_ids"][a:b], skip_special_tokens=False) for a, b in spans if any(v != -100 for v in encoded["labels"][a:b])],
                })
            if (index + 1) % 128 == 0:
                print(f"Verified assistant labels: {split} {index + 1}/{len(pool)}", flush=True)
        if not tokenized[split]:
            raise ValueError(f"No usable examples in {split}")
    train_groups = {r["group_id"] for r in used["train"]}
    eval_groups = {r["group_id"] for r in used["eval"]}
    if train_groups & eval_groups and not cfg.overfit_debug:
        raise RuntimeError("Training/validation prompt groups overlap")
    data_summary = {
        "source_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "eval_source_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest() if eval_path else None,
        "task_split_manifest_sha256": hashlib.sha256(task_manifest_path.read_bytes()).hexdigest() if task_manifest_path else None,
        "rl_holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest() if holdout_path else None,
        "splits": {name: {
            "examples": len(examples), "groups": len({r['group_id'] for r in used[name]}),
            "total_tokens": sum(len(e["input_ids"]) for e in examples),
            "supervised_tokens": sum(sum(v != -100 for v in e["labels"]) for e in examples),
            "max_tokens": max(len(e["input_ids"]) for e in examples),
        } for name, examples in tokenized.items()},
        "dropped": dropped, "train_eval_group_overlap": len(train_groups & eval_groups),
        "overfit_debug": cfg.overfit_debug, "train_order": cfg.train_order,
        "rl_holdout_checked": holdout_path is not None,
        "tokenized_fingerprint": digest(tokenized),
        "loss_mask": "assistant responses and turn endings only; all other labels are -100",
        "example_normalized_loss":cfg.example_normalized_loss,
    }
    if cfg.example_normalized_loss:
        data_summary['objective_category_mass']={split:{category:sum(r['loss_weight'] for r in rows if r['category']==category)/len(rows) for category in sorted({r['category'] for r in rows})} for split,rows in used.items()}
    manifest = {name: [{"id": r["id"], "group_id": r["group_id"], **({"task_id": r["task_id"]} if "task_id" in r else {})} for r in pool] for name, pool in used.items()}
    if cfg.train_order == "tool_call_count":
        from tau2_grpo.sft_data import tool_call_count
        counts = [tool_call_count(r) for r in used["train"]]
        if counts != sorted(counts):
            raise RuntimeError("Tool-call curriculum order was not preserved")
        (run_dir / "curriculum_order.json").write_text(json.dumps([
            {"id": r["id"], "tool_calls": count} for r, count in zip(used["train"], counts, strict=True)
        ], indent=2))
    previous_runtime = None
    if resume:
        previous_runtime = json.loads((run_dir / "runtime.json").read_text())
        if previous_runtime["resolved_model_revision"] != revision:
            raise ValueError("Base model revision changed since the saved run")
        old_data = json.loads((run_dir / "data_summary.json").read_text())
        if old_data["tokenized_fingerprint"] != data_summary["tokenized_fingerprint"]:
            raise ValueError("Resume requires identical prepared training and validation data")
        old_cfg = json.loads((run_dir / "config.json").read_text())
        if old_cfg.get("train_order", "random") != cfg.train_order:
            raise ValueError("Resume setting changed: train_order")
        for key in ("model_id", "model_revision", "initial_adapter", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules", "wandb_project", "wandb_entity", "wandb_mode"):
            if old_cfg.get(key) != settings_dict(cfg)[key]:
                raise ValueError(f"Resume setting changed: {key}")
    config_name = f"config-resume-{int(time.time())}.json" if resume else "config.json"
    (run_dir / config_name).write_text(json.dumps(settings_dict(cfg), indent=2))
    (run_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2))
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "mask_preview.json").write_text(json.dumps(mask_previews, indent=2))
    print(json.dumps(data_summary, indent=2), flush=True)
    if args.prepare_only:
        print("Data preparation passed. No model weights loaded or training started.")
        return

    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer

    from tau2_grpo.callbacks import AdapterUpdateAudit
    from tau2_grpo.sft_loss import configure_chunked_loss, preserve_generation_signature
    from tau2_grpo.sft_backends import SFTBackend

    backend = SFTBackend(
        cfg.backend, records=used["train"], encoded=tokenized["train"],
        tokenizer=tokenizer, max_length=cfg.max_length, run_dir=run_dir, config=cfg,
    )
    print(f"SFT backend: {cfg.backend}", flush=True)

    if cfg.loss_type == "chunked_nll":
        configure_chunked_loss(cfg.loss_chunk_size, cfg.loss_projection_dtype)

    if not cfg.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use use_cpu=true and precision=fp32 for CPU checks")
    if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; choose fp16 or fp32")
    if not cfg.use_cpu and torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    set_seed(cfg.seed)
    name = cfg.wandb_run_name or f"tiny-aya-sft-{run_dir.name}"
    tracking = wandb.init(
        project=cfg.wandb_project, entity=cfg.wandb_entity, name=name,
        mode=cfg.wandb_mode, tags=cfg.wandb_tags, job_type="sft",
        dir=str(run_dir), config={**settings_dict(cfg), "data": data_summary},
        id=previous_runtime["wandb_run_id"] if previous_runtime else None,
        resume="must" if previous_runtime and cfg.wandb_mode == "online" else None,
        settings=wandb.Settings(init_timeout=60),
    )
    if tracking is None or tracking.disabled:
        raise RuntimeError("W&B failed to initialize; training will not proceed without logging")
    success = False
    try:
        runtime = {
            "model_id": cfg.model_id, "resolved_model_revision": revision,
            "wandb_run_id": tracking.id, "wandb_url": tracking.url,
            "wandb_mode": cfg.wandb_mode,
            "packages": {p: version(p) for p in ("torch", "transformers", "trl", "peft", "datasets", "wandb")},
            "device": "cpu" if cfg.use_cpu else torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **backend.metadata,
        }
        (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
        print(f"W&B run: {tracking.url or tracking.id} ({cfg.wandb_mode})", flush=True)
        tracking.log({"data/train_examples": len(used["train"]), "data/eval_examples": len(used["eval"])})
        tracking.log({"data/mask_preview": wandb.Table(
            columns=["id", "total_tokens", "supervised_tokens", "assistant_targets"],
            data=[[p["id"], p["total_tokens"], p["supervised_tokens"], "\n\n".join(p["assistant_targets"])] for p in mask_previews],
        )})
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.precision]
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=revision, dtype=dtype, config=base_config,
            attn_implementation=cfg.attn_implementation,
        )
        model.config.use_cache = False
        if cfg.initial_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, str(resolve_path(cfg.initial_adapter)), is_trainable=True,
            )
        if cfg.mlp_chunk_size:
            from tau2_grpo.mlp_chunking import enable_mlp_chunking
            chunked_layers = enable_mlp_chunking(model, cfg.mlp_chunk_size)
            print(f"MLP memory chunking: {chunked_layers} layers, {cfg.mlp_chunk_size} tokens per chunk", flush=True)
        steps_per_epoch = math.ceil(len(used["train"]) / (cfg.batch_size * cfg.gradient_accumulation_steps))
        planned_steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(steps_per_epoch * cfg.num_train_epochs)
        warmup_steps = math.ceil(planned_steps * cfg.warmup_ratio)
        training_args = SFTConfig(
            output_dir=str(run_dir), run_name=name,
            num_train_epochs=cfg.num_train_epochs, max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate, warmup_steps=warmup_steps,
            lr_scheduler_type=cfg.lr_scheduler_type, weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm, optim=cfg.optim,
            bf16=cfg.precision == "bf16", fp16=cfg.precision == "fp16", use_cpu=cfg.use_cpu,
            gradient_checkpointing=cfg.gradient_checkpointing,
            activation_offloading=cfg.activation_offloading,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=cfg.max_length, packing=False, loss_type=cfg.loss_type,
            # Labels are already verified and masked. Never rebuild or truncate them.
            dataset_kwargs={"skip_prepare_dataset": True},
            assistant_only_loss=False, completion_only_loss=False,
            logging_strategy="steps", logging_steps=cfg.logging_steps, logging_first_step=True,
            eval_strategy="steps", eval_steps=cfg.eval_steps,
            save_strategy="steps", save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model="eval_loss", greater_is_better=False,
            prediction_loss_only=True, report_to=["wandb"], seed=cfg.seed, data_seed=cfg.seed,
            remove_unused_columns=not cfg.example_normalized_loss,
            train_sampling_strategy="sequential" if cfg.train_order == "tool_call_count" else "random",
            dataloader_num_workers=0, dataloader_pin_memory=not cfg.use_cpu,
            push_to_hub=False,
        )
        audit = AdapterUpdateAudit()
        extra_callbacks=[]
        if extension:
            from tau2_grpo.context_extension import checkpoint_callback
            extra_callbacks.append(checkpoint_callback(base_config,extension))
        def build_trainer(train_rows):
            # Reset immediately before LoRA initialization for backend parity.
            set_seed(cfg.seed)
            trainer_type=SFTTrainer
            if cfg.example_normalized_loss:
                from tau2_grpo.sft_weighted import trainer_class
                trainer_type=trainer_class()
            trainer=trainer_type(
                model=model, args=training_args, processing_class=tokenizer,
                train_dataset=Dataset.from_list(train_rows),
                eval_dataset=Dataset.from_list(tokenized["eval"]),
                peft_config=None if cfg.initial_adapter else LoraConfig(
                    r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=cfg.lora_target_modules, bias="none", task_type="CAUSAL_LM",
                    revision=revision,
                ), callbacks=[audit]+extra_callbacks,
            )
            trainer.weighted_chunk_size=cfg.loss_chunk_size
            return trainer

        def on_ready(trainer):
            preserve_generation_signature(trainer.model)
            if not any(type(c).__name__ == "WandbCallback" for c in trainer.callback_handler.callbacks):
                raise RuntimeError("Trainer did not install its W&B callback")
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            tracking.summary["trainable_parameters"] = trainable
            print(f"Trainable LoRA parameters: {trainable}", flush=True)
            if not cfg.use_cpu:
                torch.cuda.reset_peak_memory_stats()
                tracking.summary["gpu_free_gib_before_training"] = torch.cuda.mem_get_info()[0] / 2**30
            if cfg.evaluate_before_training and not resume:
                baseline_metrics = trainer.evaluate(metric_key_prefix="baseline_eval")
                (run_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
                tracking.log({f"baseline/{key}": value for key, value in baseline_metrics.items()})

        result = backend.train(
            build_trainer=build_trainer, on_ready=on_ready,
            resume=str(resume) if resume else None,
        )
        trainer = backend.trainer
        tracking.summary.update(backend.metadata)
        metrics = trainer.evaluate()
        changed = sum(
            not torch.equal(audit.before[n], p.detach().float().cpu())
            for n, p in trainer.model.named_parameters() if p.requires_grad
        )
        if not changed or not math.isfinite(float(result.training_loss)):
            raise RuntimeError("Training did not produce a finite loss and a verified LoRA update")
        adapter_dir = run_dir / "final_adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        if extension:save_extension(adapter_dir,base_config,extension)
        trainer.save_state()
        final_metrics = {**result.metrics, **metrics, "changed_adapter_tensors": changed}
        if not cfg.use_cpu:
            final_metrics.update(
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
        (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
        (run_dir / "trainer_log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        tracking.summary.update(final_metrics)
        tracking.summary["final_adapter"] = str(adapter_dir)
        samples = []
        trainer.model.eval()
        # The chunked-loss forward has a generic signature, so generate() can
        # miss this capability and allocate logits for the entire prompt.
        # Last-token-only logits give the same autoregressive output, with much
        # lower peak memory for Tiny Aya's 262k-token vocabulary.
        base_forward = trainer.model.get_base_model().__class__.forward
        generation_options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(base_forward).parameters else {}
        for record in used["eval"][:cfg.log_sample_count]:
            index = next(i for i, m in enumerate(record["messages"]) if m["role"] == "assistant")
            prompt = record["messages"][:index]
            inputs = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(trainer.accelerator.device)
            available = min(cfg.max_length, base_config.max_position_embeddings) - inputs["input_ids"].shape[1]
            if available <= 0:
                continue
            with torch.inference_mode():
                generated = trainer.model.generate(
                    **inputs, max_new_tokens=min(cfg.sample_max_new_tokens, available),
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    **generation_options,
                )[0, inputs["input_ids"].shape[1]:]
            samples.append({
                "id": record["id"], "prompt": prompt,
                "reference": record["messages"][index]["content"],
                "generated": tokenizer.decode(generated, skip_special_tokens=True),
            })
        (run_dir / "generated_samples.json").write_text(json.dumps(samples, indent=2))
        if samples:
            tracking.log({"samples/tool_calls": wandb.Table(
                columns=["id", "prompt", "reference", "generated"],
                data=[[s["id"], json.dumps(s["prompt"]), s["reference"], s["generated"]] for s in samples],
            )})
        print(f"Saved adapter: {adapter_dir}", flush=True)
        print(json.dumps(final_metrics, indent=2), flush=True)
        success = True
    finally:
        tracking.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
