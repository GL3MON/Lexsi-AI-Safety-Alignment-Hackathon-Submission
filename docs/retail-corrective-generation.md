# Corrective retail SFT regeneration

The new campaign uses the existing Qwen3.5-4B endpoint on MIG2 (port 8004) for
both customer and teacher, with separate conversation contexts. The previous
SFT dataset and checkpoints are unchanged. No training is launched by this job.

The existing task manifest reserves 40 SFT tasks and 10 validation tasks. Each
SFT task receives three generations: identity clarification, one correctable
identity lookup failure, and confirmation/execution. Validation profiles are
balanced across the ten tasks. The original task goals and conditional requests
are preserved. The recovery profile creates one typo from a known identity field
and then supplies the original correct value; it never invents order IDs.

Both actor prompts are snapshotted in the new collection directory. Changes to
them require a new output version; actor-prompt code changes stop the collector
before another child episode. Reward-zero traces are retained as requested and
no model policy reviewer runs. Raw traces, rewards, prompt hashes and profiles
are saved for inspection; prompt instructions do not guarantee every generated
example will be correct.

```bash
scripts/run_corrective_retail_pipeline.sh
# Resume an interrupted campaign with the same frozen prompts:
scripts/run_corrective_retail_pipeline.sh --resume
```

Outputs:

- `outputs/retail-data/retail-sft-corrective-v2/progress.json`: live counts.
- `outputs/retail-data/retail-sft-corrective-v2/runs/`: full raw JSON and TXT traces.
- `outputs/retail-data/retail-sft-corrective-v2/exports-all-generations/`: full candidates.
- `outputs/retail-data/retail-sft-corrective-v2-final/`: final context-prepared data,
  produced after collection completes using the MIG2 summarizer.
- `outputs/logs/retail-corrective-v2-pipeline.log`: coordinator log.

The collector's intermediate `sft.jsonl` may contain 8K prefix previews. The final
preparation reads the complete saved candidate episodes, preserves assistant
targets, and masks repeated context; use the final directory for training after
checking its report. Infrastructure failures remain errors rather than fabricated
reward-zero examples. Final preparation quarantines incomplete/failed summaries.

The confirmation-sensitive task-92 pilot passed its DB check with reward 1. The
final task-95 recovery pilot also scored 1, with the intended failed ZIP lookup,
explicit correction, successful authentication and subsequent exchange actions.
All 53 repository tests passed before the full campaign was started.

Collection is now split between MIG2/port 8004 and MIG0/port 8005. The two
collectors own disjoint task sets; the first two completed generations on MIG2
were retained. The combined live output is
`outputs/retail-data/retail-sft-corrective-parallel-v2/`, including `workers.json`,
`worker-status.json`, and `progress.json`. MIG0's worker writes to
`outputs/retail-data/retail-sft-corrective-v2-mig0/`. Both use identical frozen
actor prompt hashes and the three per-task profiles. Start/resume this layout
using `scripts/run_parallel_corrective_retail_sft.sh`; its final preparation uses
both endpoints. The main log is `outputs/logs/retail-corrective-parallel-v2.log`.
`corrective-coverage-plan.json` distinguishes the five instructed behaviors and
successful pilot evidence from unverified full-corpus behavioral coverage.
