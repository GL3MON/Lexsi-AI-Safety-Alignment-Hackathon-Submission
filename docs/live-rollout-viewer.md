# Live RL rollout viewer

Open **http://localhost:8766** while the server runs. It binds to `127.0.0.1` by default. On a remote machine, forward port 8766 through your editor or SSH.

Start from the project directory:

```bash
.venv/bin/python -u scripts/serve_rollouts.py
```

The browser refreshes every five seconds. A read-only background worker mirrors the newest job's episode JSON, group summaries, and validation results from its private HF bucket every 15 seconds. It does not load models, use a GPU, alter training, or upload anything. HF credentials stay on the server. The existing job watcher supplies logs/status; the viewer refreshes stale status independently.

Select the current run or the previous short pilot in the run menu. Group reward bars filter the episode list. Search by task/scenario and filter by phase, success, failure, in-progress, or tied groups. Select an episode to inspect customer, assistant, and tool messages; tool JSON is formatted separately. System prompts and evaluation checks are expandable. Download the conversation as `.txt` or inspect raw JSON.

Rewards and advantages remain pending until the corresponding group is scored. Tied groups have zero relative advantage and skip the update; the viewer displays zero explicitly even when the training artifact records `advantages: null`. Displayed summary totals refer to scored training groups. Validation results are separate; neither is native Tau2 benchmark success. Text-only historical caches do not include per-action token metadata. Response updates arrive at file-sync boundaries, not token by token.

Options:

```bash
# Cached files only; no HF connection
.venv/bin/python scripts/serve_rollouts.py --no-sync --port 8766

# Choose job metadata; repeat --job to include historical runs
.venv/bin/python scripts/serve_rollouts.py --job hf_jobs/short_bank_rl/job.json
```

Implementation: `scripts/serve_rollouts.py` and `tools/rollout_viewer.html`.
Log: `outputs/logs/rollout-viewer.log` when started with the output redirection used in this session.

Verification: five backend tests cover advantage/reward handling, role boundaries, token-payload trimming, partial JSON, and path traversal. Headless Chrome rendered live episode/message cards without a displayed error.
