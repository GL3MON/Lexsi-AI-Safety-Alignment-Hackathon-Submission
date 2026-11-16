# Development and maintenance

## Source map

| File | Main symbols / responsibility |
| --- | --- |
| `src/tau2_grpo/__init__.py` | Package identity; no workflow execution. |
| `src/tau2_grpo/config.py` | `ExperimentConfig`, `load_config`; defaults, type/range checks. |
| `src/tau2_grpo/cli.py` | `main`; argparse, path validation, doctor and workflow launch. |
| `src/tau2_grpo/rollout.py` | `configure`, phase/event helpers, `Tau2SoloRollout`, `terminal_task_reward`, cleanup and `BASE_PROMPT`. |
| `src/tau2_grpo/callbacks.py` | `AdapterUpdateAudit`; snapshot actual trainable adapter initialization. |
| `src/tau2_grpo/experiment.py` | Fourteen ordered stages, dataset creation, trainer factory, evaluation summaries, retraining callback, gate and reload. |
| `tests/test_config.py` | Preset validity, invalid settings, source syntax parsing. |
| `scripts/setup.sh` | Environment creation/reuse and installation. |
| `requirements-gpu.txt` | Direct GPU/AgentTune contract. |
| `requirements-local.lock` | Machine-specific freeze including editable source paths. |
| `configs/local.json`, `configs/small.json` | Editable experiment overrides. |
| `Makefile` | Setup, doctor, smoke, train, small and test shortcuts. |

The experiment's module-level ordering is intentional. Use the CLI; avoid
importing `experiment.py` in tooling just to inspect a symbol. Heavy imports are
staged after the smoke-only early exit. Ruff E402 is explicitly disabled for that
workflow file because moving all imports to the top would change smoke behavior.

## Verification commands and their limits

```bash
make test
bash -n scripts/setup.sh
uvx ruff check --select E4,E7,E9,F,I src tests
uvx ruff format --check src tests
uv pip check --python .venv/bin/python
make doctor
make smoke
```

`make test` uses system `python3` with `PYTHONPATH=src`; it requires no model, GPU
or tau2 installation. There are three test methods: accepted presets, representative
invalid settings, and AST parsing of source files. They do not test GRPO gradients,
AgentTune lifecycle, model loading, gate behavior, partition correctness at runtime
or statistical performance. `uvx ruff` may download tooling if not cached.

`make smoke` is a real integration check: partition setup, first fitting task,
policy/ticket/tool catalog construction, customer lookup, explicit done, evaluated
terminal reward and saved episode. It is scripted and never samples an LLM.
Training also runs it before loading a model. Full training requires the pinned
GPU stack and enough memory and remains separately recorded in `VALIDATION.md`.

## Extending the project

For a new JSON setting, add a typed default and suitable validation to
`ExperimentConfig`/`load_config`, wire it into the workflow, update presets if
needed and document it in the configuration table. Add tests when validation
or behavior changes meaningfully.

For a model change, check native tool calling, effective chat template, model
context support, dtype and LoRA target module names. Passing a different
`model_id` does not establish compatibility. Inspect saved trajectories before
interpreting training metrics.

For a domain change, replace the telecom split loader and prompt, adapt the smoke
lookup/phone extraction, verify solo Gym support and tool schemas, and preserve
train/validation/test separation. The validator currently rejects other domains
on purpose. This is not a generic switch-domain trainer.

For a dependency upgrade, evaluate the AgentTune factory/lifecycle, TRL environment
method exposure and reward callback contracts, chat-template behavior, completion
budget semantics and PEFT save/load format. The explicit pins and runtime checks
must change together. Update the local freeze only after validating the resulting
environment; do not remove checks merely to make imports proceed.

For repeated closed-loop rounds, add a deliberate state/cancellation protocol,
fresh fitting-failure collection and development-validation policy. Keep final test
reserved until model selection is complete. Current phase globals assume exactly
one workflow at a time. A multi-process implementation needs explicit per-run and
per-rollout state rather than reusing these globals.

## Current scope and limitations

- One telecom solo cycle on one visible GPU in one process.
- Fresh runs only; saved optimizer state is not exposed through a resume command.
- No serving endpoint, Hub publishing or external deployment action.
- Terminal task rewards only; substring communication scoring; no reward judge.
- Small fitting/validation/test samples, stochastic rollouts and no significance test.
- Shared phase/trainer state; no concurrent experiments in one interpreter.
- Four-hour source-defined wait and no cooperative background-worker cancellation.
- Local source layout and editable installs; path discovery and freeze are not a
  portable wheel/distribution contract.
- Runtime records tau2 Git HEAD but does not snapshot dirty data/code or standalone
  project source; archive those separately for reproduction.
- Final report precedes the reload probe; successful report creation is not a full
  pipeline-completion marker.

The `.gitignore` excludes `.venv`, Python caches, egg metadata, `.env`, Ruff cache
and output contents except `.gitkeep`. Runs and model artifacts need separate
archival if they are to be retained beyond this workstation.
