# Artifacts and results

Each `smoke` or `train` invocation creates a new `run-<10 hex characters>` beneath
`--output`. It prints that path at startup. `config` and `doctor` do not create run
directories. Episode filenames use their own rollout UUIDs.

## File inventory

Files appear as the run reaches the corresponding stage. Later files are absent
in smoke-only, interrupted or failed runs.

| Path within a run | Contents / stage |
| --- | --- |
| `experiment_config.json` | Fully merged experiment config, written immediately. |
| `split_manifest.json` | Full fitting, validation and official-test partitions plus selected samples. |
| `episodes.jsonl` | One compact JSON record per finalized episode across phases. |
| `episode-<uuid>.json` | Full saved simulation plus wrapper action history and forced-stop flag. |
| `completion-<uuid>.txt` | Decoded model rollout with special tokens and tool responses, saved when scoring begins. |
| `completion-<uuid>.json` | Phase, task and rollout IDs, parsed conversation messages, and decoded raw text. Includes tool calls rejected before reaching Gym. |
| `runtime.json` | Package versions, base model/revision, GPU/dtype, scoring, AgentTune pin and tau2 Git HEAD; written after baseline trainer setup. |
| `collect_fit-metrics.json` | TRL evaluation metrics for baseline fitting scan. |
| `failed_task_dataset.jsonl` | Initial dataset of distinct failed fitting task IDs and generic prompts. |
| `failed_episode_index.json` | Compact baseline failure events with evidence paths. |
| `buffer_records.jsonl` | Serialized negative-only AgentTune failure records. |
| `tau2_audit.jsonl` | Baseline fitting audit entries, PASS only for reward exactly 1.0. |
| `baseline_validation-metrics.json` | Baseline fixed-validation metrics. |
| `drained_grpo_dataset.jsonl` | Dataset reconstructed from the actual drained failure buffer. |
| `checkpoints/` | TRL output/checkpoints and saved state; no CLI resume integration exists. |
| `training_metrics.json` | AgentTune stats, changed-weight flag, changed tensor count and selected task IDs. |
| `trainer_log_history.json` | Trainer's log-history list. |
| `candidate_adapter/adapter_config.json` | PEFT adapter configuration. |
| `candidate_adapter/adapter_model*` | Saved adapter weights, safetensors or binary. |
| `candidate_adapter/effective_chat_template.json` | Effective template retained from training/tokenizer. |
| `candidate_adapter/` additional files | Files emitted by AgentTune save, potentially tokenizer, training config/stats and provenance; inspect actual contents. |
| `candidate_validation-metrics.json` | Candidate fixed-validation metrics. |
| `deployment_manifest.json` | Initial baseline selection, optionally atomically replaced after gate approval. |
| `closed_loop_cycle.json` | Serialized AgentTune cycle, training result, gate decision and apply result. |
| `final_test_baseline-metrics.json` | Test evaluation with adapter disabled. |
| `final_test_candidate-metrics.json` | Test evaluation with candidate adapter enabled. |
| `final_report.json` | Final test summaries, selection/gate info and adapter path; written before reload check. |

Raw TRL metrics need not use the same weighting as the explicit final task summaries.
Use `mean_task_reward` in those summaries to compare equal-weight task means.

## Episode schema

An `episodes.jsonl` record contains:

```json
{
  "phase": "cpu_smoke",
  "task_id": "<actual tau2 task ID>",
  "rollout_id": "<UUID>",
  "task_reward": 0.0,
  "actions": 2,
  "forced_stop": false,
  "simulation_path": "/absolute/path/to/episode-<UUID>.json"
}
```

This is an illustrative shape, not a training result. `actions` is a count,
including malformed argument attempts. The full episode file has `phase`,
`actions` (the action/observation list), `forced_stop` and `simulation` (τ-bench's
serialized run including `reward_info`). Reward comes from
`simulation.reward_info.reward`. A forced stop means the wrapper finalized an
unfinished episode; it is not automatically a zero reward.

## Phase names

| Phase | Role |
| --- | --- |
| `cpu_smoke` | Scripted tool lookup and explicit finish; every training run also starts with this check. |
| `collect_fit` | Baseline failure selection on fitting sample. |
| `baseline_validation` | Cached baseline scores for fixed validation. |
| `grpo_train` | Fresh on-policy rollouts during optimization. |
| `candidate_validation` | Candidate scores for the gate. |
| `final_test_baseline` | Original base policy, adapter disabled. |
| `final_test_candidate` | Candidate policy, adapter enabled. |

Do not mix smoke, fitting, training, validation and test episodes when computing
an experiment score.

## Inspect a run

Set the path explicitly using the directory printed by the command:

```bash
RUN_DIR=outputs/run-db6b8ef791
.venv/bin/python -m json.tool "$RUN_DIR/experiment_config.json"
.venv/bin/python -m json.tool "$RUN_DIR/split_manifest.json"
```

This command summarizes available episode events without requiring final files:

```bash
.venv/bin/python - "$RUN_DIR" <<'PYCODE'
import json
import sys
from collections import defaultdict
from pathlib import Path
run = Path(sys.argv[1])
groups = defaultdict(list)
for line in (run / "episodes.jsonl").read_text().splitlines():
    event = json.loads(line)
    groups[event["phase"]].append(event)
for phase, events in groups.items():
    print(phase, "episodes=", len(events),
          "tasks=", len({e["task_id"] for e in events}),
          "forced_stops=", sum(e["forced_stop"] for e in events))
PYCODE
```

For a training run that reached selection/reporting:

```bash
.venv/bin/python -m json.tool "$RUN_DIR/training_metrics.json"
.venv/bin/python -m json.tool "$RUN_DIR/closed_loop_cycle.json"
.venv/bin/python -m json.tool "$RUN_DIR/deployment_manifest.json"
.venv/bin/python -m json.tool "$RUN_DIR/final_report.json"
```

The known smoke directory above intentionally has no training or final-report
files. Substitute the training run ID before running those commands.

## Selection and evidence

The initial deployment manifest contains model ID/revision,
`selected_adapter: null` and `local_manifest_only: true`. Approval sets
`selected_adapter` to the saved adapter path and adds AgentTune revision/protocol.
A `kept_old` cycle can still have a real trained candidate, complete test results
and a successful adapter reload. It means the candidate was not selected.

Read `weights_changed`, `gate_decision`, `validation_approved`, `loop_applied` and
both final test summaries together. Changed weights alone do not establish better
performance. A validation improvement on one or four sampled tasks does not
establish statistical significance or benchmark-wide generalization.

`final_report.json` includes `protocol`, `baseline_test`, `candidate_test`,
`validation_approved`, `deployment_manifest`, `adapter`, `weights_changed`,
`training_failed_tasks`, `agenttune_git_revision`, `loop_applied` and
`gate_decision`. Each test summary contains `tasks`, `episodes`,
`mean_task_reward` and `per_task_mean`.

Archive the run, command/config, shell log, source checkout state and installed
environment snapshot together. `runtime.json` records tau2 Git HEAD but not dirty
working-tree contents or a hash of this standalone project. Exact reproduction
requires preserving those sources separately. Several artifact paths are absolute;
moving the directory does not rewrite them. Output files are ignored by Git and
are retained locally until you explicitly archive or remove them.
