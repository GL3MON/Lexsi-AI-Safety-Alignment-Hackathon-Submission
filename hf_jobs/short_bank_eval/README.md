# Checkpoint 28 validation at temperature 0.7

Inference only: one sampled trajectory for each of the same 30 fixed validation cases used in the completed short-bank RL run. Initial adapter is the final policy checkpoint 28, SHA256 `fa0040fc02fd474ca22c620e883aac5e0daa5542e2b28f4d344f3b1df0978498`. No optimizer, reference adapter, or policy updates are created.

Settings: Transformers BF16, temperature 0.7, top-p 1, top-k 0, repetition penalty 1, seed 42 per case with per-turn offsets, 192 response tokens, at most four actions, 8K context. Original SFT system and Jinja template retained. The only intended decoding change from final greedy validation is sampling at temperature 0.7. Token-probability replay is omitted for inference; environment and completion validation are unchanged.

The evaluation asserts that its case IDs match the saved final greedy summary exactly. It reports individual results alongside those original greedy results. A single sampled attempt is not pass@k or evidence of an average gain across seeds. Native Tau2 rewards are not evaluated. Infrastructure failures remain explicit, rather than silently counted as model failures.

L40S job, 20-minute hard cap, maximum budget reservation $0.64. Reuses the existing private bucket archive and checkpoint; no Hub output commits. Progress, individual JSON/text rollouts, final summary and W&B metadata are saved under the job's private bucket prefix. See `job.json` for current job details and `outputs/hf-jobs/<job_id>/` for watcher outputs.

The live viewer supports this job via:

```bash
.venv/bin/python scripts/serve_rollouts.py --job hf_jobs/short_bank_eval/job.json --job hf_jobs/short_bank_rl/job.json --job hf_jobs/short_retail_rl/job.json
```

Use its phase selector to select `short-temp07` or all phases. The 17/30 greedy baseline is developmental validation, not a native benchmark result.
