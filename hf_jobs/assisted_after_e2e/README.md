# Assisted RL from end-to-end SFT

Active initializer: `retail-e2e-sft-20261115T212705/training/run-ad72681cc0/checkpoint-70`. Parent SFT job reached its timeout; checkpoints60and70were saved. Checkpoint70 is the latest complete checkpoint and has validation loss0.3958349824. It is cached locally at `outputs/rl-initialization/retail-e2e-checkpoint-70` with provenance and SHA256 `3d9f1b681b8a77465714fcdc3fc99494b27a3e28631eb008a55b3ff85d96fe64`.

The user explicitly authorized recovery from the latest saved checkpoint and direct RL launch. CLI `--latest-sft-checkpoint` validates inactive parent status, checkpoint files, trainer step, safetensors structure, tokenizer EOS6and exact chat template. The automatic dependency watcher still requires completed SFT and rollout markers and cannot use this override by itself. No fake completion marker is written.

Run:30groups ×16rollouts, success-gated finish→find→unassisted assistance,20% easier-level replay. Original reserved RL IDs only; demonstrated cancellation/return/shipping workflows first. No standalone grounded-call bonus. Fresh optimizer and frozen reference both initialize from checkpoint70. Native smoke and unassisted validation precede updates; validation repeats every10groups.

L40S,120-minute cap,$3.64reservation, approximately$2.55remaining tracked account budget. Viewer and monitoring log point to this new run. The standalone post-SFT rollout did not run before the parent timeout; this is explicitly recorded as checkpoint recovery, not a successfully completed previous chain.
