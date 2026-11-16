# Retail SFT

Launch from the repository:

```bash
scripts/run_retail_sft.sh
```

The launcher defaults to MIG0 (`MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04`).
It does not stop servers automatically. MIG2 is left available for inference.
Override the training device with `RETAIL_SFT_GPU` if needed.

`configs/sft-retail.json` selects the checkpoint-525 Aya model, AgentTune backend,
184 finalized training examples and 10 validation examples, 8,192-token context,
batch size 1, gradient accumulation 8, two epochs, BF16 LoRA, activation
offloading and online W&B logging in project `tau2-retail`.

Only assistant messages listed in each example's `loss_message_indices` receive
loss, including their turn endings. System instructions, customer messages,
tool responses, summaries, repeated assistant context, and padding are masked
with `-100`. Packing and truncation are disabled. Overlength examples fail.
AgentTune projections must preserve the verified token IDs and labels exactly.

Audit the final data without loading model weights:

```bash
.venv/bin/python scripts/audit_retail_sft.py
scripts/run_retail_sft.sh --prepare-only
```

The audit writes `outputs/retail-data/retail-sft-final/training-audit.json`.
The selected data passed with 112 complete source trajectories; nine source
trajectories that failed summary validation remain excluded. The original
preparation report retains those failures. All original assistant targets in
the selected data are preserved and supervised exactly once.

Training writes a new directory under `outputs/sft-retail/` containing config,
data manifest, mask previews, runtime information and W&B URL, checkpoints,
backend audit, and the final adapter. The current detached launch logs to
`outputs/logs/retail-sft-training.log`; its PID is recorded in
`outputs/logs/retail-sft-training-process.json`.

For example, CLI settings override JSON settings:

```bash
scripts/run_retail_sft.sh --set batch_size=1 --set gradient_accumulation_steps=8
```

## Database-check follow-up

The training trajectories are retained intentionally; database-check failures are
tracked for follow-up without filtering SFT or changing task partitions. See
`outputs/retail-data/retail-sft-final/db-check-tracker.md` and its JSON/CSV
companions. There are 47 failing source generations across 25 SFT task IDs; a
flagged task may also have passing generations. Missing DB checks are tracked
separately. Regenerate the tracker with:

```bash
.venv/bin/python scripts/track_sft_db_checks.py
```

The function-call loop audit is
`outputs/retail-data/retail-sft-final/tool-call-loop-audit.txt`. It scans the 107
actual training source episodes (805 supervised calls), excluding 52 masked
context call repetitions. One task-95 generation, training JSONL line 62, contains
10 failed order lookups with customer dialogue and varying nonexistent order IDs.
All ten calls are supervised. The other task-95 generations are tracked separately.
This flag does not remove or modify the training example. Reproduce with:

```bash
.venv/bin/python scripts/audit_retail_sft_loops.py
```
