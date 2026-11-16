# Hugging Face retail RL job — setup only

No HF compute job is launched by preparing these files. The launcher is a dry run unless `--submit` is supplied. Payload upload also requires an explicit flag.

The initial plan uses one 48 GB L40S for a frozen Qwen3.5-4B vLLM customer/summarizer/judge and a Transformers/PEFT Aya policy trainer, plus a colocated Aya vLLM engine. Eight spawned CPU environment workers share the actor engine; they do not load eight model copies. They use isolated Python environments. It starts from the preserved retail SFT checkpoint 100 and runs two groups of eight rollouts, at most ten actor actions each, on the original RL task partition. Rewards are additive Tau2 checks plus grounded-call shaping and repeated-failure penalties. W&B remains online and includes per-rollout advantages and conversations.

The job uses the original task goals, not the supported-only diagnostic customer override. It does not modify the official task files or learn from validation task 63.

## Preparation commands

Build a credential-free payload locally:

```bash
.venv/bin/python hf_jobs/retail_rl/prepare.py
```

Explicitly upload source, data, and the adapter to private HF datasets, without running compute:

```bash
.venv/bin/python hf_jobs/retail_rl/prepare.py --upload
```

Read current prices and print the plan without submitting:

```bash
.venv/bin/python hf_jobs/retail_rl/launch.py
```

Only when a paid run is authorized, explicitly submit:

```bash
.venv/bin/python hf_jobs/retail_rl/launch.py --submit
```

The default timeout is 45 minutes. At the queried $1.80/hour L40S rate, that is about $1.35 before the conservative rounding margin. Prices are rechecked at launch. A local ledger reserves each launch's timeout-based maximum against a $15 spendable ceiling, protecting $5 of the authorized $20. Automatic retries are not enabled. This reservation is a guard for this launcher, not an account-wide provider billing limit.

HF and W&B credentials are read at submission and passed through encrypted job secrets, never packed in the payload. Checkpoints, rollout JSON, checks, and logs are uploaded periodically and at normal process exit to a private results dataset. A provider-enforced hard timeout may interrupt the final upload; periodic upload reduces, but does not eliminate, that risk.

## Validation status

- Local reward/rollout unit checks passed (21 tests), plus four vLLM protocol/adapter/configuration tests.
- vLLM rollout setup still requires a real cloud memory/startup smoke; passing local protocol tests is not proof of GPU execution.
- The actor server returns processed sampling log probabilities and exact token IDs. Trainer replay probabilities are stored separately for importance correction. Adapter changes occur only between completed groups.
- All four public base-model weight shards were verified byte-for-byte against the local exported base.
- The preserved checkpoint-100 adapter has source provenance and a SHA-256 record.
- Bootstrap Python syntax and the payload can be checked locally without allocating GPUs.
- This cloud dependency environment, vLLM customer, and full RL optimizer path have **not yet been exercised on HF Jobs**. The bounded run is intended to verify them before any longer training.
- A tied-reward group legitimately produces no optimizer update; the smoke does not manufacture contrast rewards.

Official references: [Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration), [live pricing API](https://huggingface.co/api/jobs/hardware), and [Jobs guide](https://huggingface.co/docs/huggingface_hub/guides/jobs).


Startup compatibility fixes (2026-11-14):

- The checkpoint tokenizer was exported by Transformers 5 as `TokenizersBackend`.
  vLLM 0.17.1 uses Transformers 4. The actor server gets a separate tokenizer
  directory with the loader class aliased to `PreTrainedTokenizerFast`; tokenizer
  JSON and token definitions are unchanged. Seven copying/tool/Unicode examples
  produce identical IDs across both environments, with EOS 6 and matching PAD.
- vLLM's Cohere `compute_logits` identifies an unwrapped embedding by `.weight`,
  but `VocabParallelEmbeddingWithLoRA` also forwards that property. The pinned
  compatibility patch checks `.base_layer` instead and uses the existing unwrap
  branch. Regression checks exercise both bare and wrapped tied output heads.
- Both failed attempts stopped during actor startup before generating rollouts.
  Neither established a GPU memory failure. A full cloud smoke remains necessary.
