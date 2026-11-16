# Continued mid-turn GRPO

100 groups × 16 rollouts, Transformers batched generation on one L40S; 60/20/20 mid-turn, opening clarification, retention curriculum. Warm-start from pilot checkpoint-6; fresh AdamW and frozen checkpoint-6 reference. Learning rate 5e-6, temperature 0.7, top-p 1.0, seed 143, 8K context. Only sampled suffix actions receive loss.

114 held-out validation cases before, every 30 groups, and after training. Last two complete optimizer checkpoints retained, with JSON/TXT rollout artifacts and online W&B tables. 180-minute timeout, $5.44 maximum tracked reservation. No periodic Hub model/dataset commits.

Launch: `.venv/bin/python hf_jobs/midturn_full_rl/launch.py --submit` (refuses duplicate active jobs).
