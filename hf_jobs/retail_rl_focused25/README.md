# Focused checkpoint25 Tau2 RL trial

Initialization is the evaluated focused-mask SFT run2190c37f90/checkpoint25, not the earlier128 pilot or the all-token200 run. Adapter SHA256:2c72a97c0abd990dd87be650ace9674051b4e30dfff8269f1e0b8a4980b0a7f2. Original Salesforce SFT system text and saved retail Jinja template are preserved exactly.

Trial:2 sampled groups from the original22-task retail RL partition,8 rollouts/group,10 actor actions maximum, Transformers for Aya, BF16, temperature0.7/top_p1. Qwen3.5-4B is hosted inside the job through vLLM for customer simulation, frozen summarization and NL evaluation. Actual Tau2 checks plus grounded-argument bonus and failure penalties; no fabricated positive reward on tied groups. AlignTune owns the training loop, with W&B online rollout tables, per-rollout advantages and per-group metrics. Optimizer updates occur only for complete non-tied groups.

Prompt compatibility ignores schema titles/description formatting, accepts trained enums that narrow runtime strings, enforces the published trained schema, and hides runtime-only unadvertised tools. Argument names, types and required-field incompatibilities still fail.31 focused harness/reward/template tests pass; real local environment preflight confirms the exact prompt and22-task split.

prepare.py --upload packages source, Tau2 retail assets, config and verified adapter into private HF datasets. launch.py --submit starts a17-minute capped (configurable up to25) L40S trial within the original protected-reserve budget. This is an RL trial, not a full sweep over all22 tasks. Inspect result.json/optimizer_steps before claiming the model trained.

User update: remote output uploads are disabled by default (RL_UPLOAD_OUTPUTS=0), including periodic and final uploads. Streaming job logs and W&B remain enabled. Changes are local until an explicitly authorized future submission.
