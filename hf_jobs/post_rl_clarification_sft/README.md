# Post-RL clarification SFT with retention

One epoch on 200 clarification +100 retention demonstrations, with44 separate SFT validation conversations. Initializer: short-bank RL checkpoint28. Batch8 × accumulation2, LR1e-5, BF16, 8K context, AlignTune, online W&B. Evaluation/save every5 optimizer steps; keep2 checkpoints and load the lowest validation-loss checkpoint at the end. Expected19 optimizer steps.

The frozen corpus includes per-message loss masks, executable trajectory checks and exact label audits. The remote job verifies the corpus archive and initializer SHA256, repeats the real encoder preflight, then trains. Source code and data are supplied once through a private bucket archive. No Hub model/dataset output commits occur.

After training, `sft-result.json` records the selected adapter, SHA256, and best checkpoint; `SFT_COMPLETE` confirms successful trainer completion. The same30 greedy validation scenarios then run under `short-post-sft/`, compared to the saved17/30 checkpoint28 baseline. The dataset uses separate worlds for training and SFT validation and excludes fixed evaluation world80 and all test worlds.

Launch: `.venv/bin/python hf_jobs/post_rl_clarification_sft/launch.py --submit`. This is paid and refuses a duplicate active launch. Budget cap25 minutes / $0.79 L40S. Read-only monitor: `.venv/bin/python -u hf_jobs/post_rl_clarification_sft/watch.py`. Job metadata is `job.json`; readable post-SFT validation rollouts and logs are retrieved under `outputs/hf-jobs/<job_id>/`.

Continued RL is prepared separately in `configs/rl-retail-clarification-mix.json`. It samples two clarification groups per retention group. It must be initialized from the selected SFT output after review; this job does not launch continued RL automatically.
