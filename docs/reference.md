> Historical source notebook documentation. For executable commands and current
> paths, use the [standalone project guide](README.md). GPU execution claims in
> this reference describe intended checks; see the [validation record](../VALIDATION.md)
> for checks actually performed locally.

# τ-bench solo → AgentTune FullClosedLoop → GPU GRPO → gate → deploy

This is the **final AgentTune integration** of the GPU experiment. It follows the
[reference self-healing notebook](https://github.com/Lexsi-Labs/AgentTune/blob/d4db4c12d52342227c4113a70183d0f64aa2f345/docs/notebooks/03_self_healing_and_closed_loop.ipynb)
at commit `d4db4c12d52342227c4113a70183d0f64aa2f345`:

```text
real solo baseline failures
       ↓
TrainingExample failure records → loop.runner.submit
       ↓
FullClosedLoop buffer → RetrainingTrigger → loop.tick()
       ↓
background retrain_job: AgentTune create_agentic_trainer("grpo").train()
       ↓
saved real LoRA adapter → build_model_runner(path)
       ↓
DeploymentGate over held-out τ-bench validation task rewards
       ↓
apply_decision → local deployment manifest OR keep baseline
       ↓
paired final test evaluation and adapter reload check
```

The reference uses real DPO training over sentiment examples; **this version replaces
the injected training job with multi-turn, on-policy GRPO over failed telecom tasks**.
It retains AgentTune's actual `FullClosedLoop`, buffer, trigger, background retrainer,
gate decision, and apply/deploy callback. No mock training or synthetic success labels
are used in the experiment.

Two deliberate adaptations are documented below: failures are submitted directly
(as in the reference) rather than passed through its optional LLM correction generator;
and the gate scores all fixed validation tasks instead of reconstructing only past
successful pipelines. The gate measures actual terminal τ-bench rewards.

**Run on your NVIDIA GPU machine.** GPU training is not executed in the authoring
environment. CPU Gym checks and source-level integration checks are a separate scope
of validation; they do not establish a GPU-tested run or a learning gain.

## 1. Prepare an isolated GPU environment

Use Linux, Python 3.12, an NVIDIA driver compatible with your CUDA-enabled PyTorch
wheel, and a clean checkout of **this τ-bench revision** containing this notebook.
Start with **one visible GPU**. The notebook checks CUDA and refuses CPU training.
The 1.7B default, long telecom prompts, four generations, and LoRA are intended for a
GPU with substantial memory; **48 GB is a conservative starting target**, not a
measured memory requirement. On a smaller GPU try the 0.6B model and group size 2.
Long policy/tool context and rollout length also affect memory.

Run these commands in the repository root **on the GPU machine**:

```bash
uv venv --python 3.12 .venv-grpo
uv pip install --python .venv-grpo/bin/python -e '.[gym]' \
  'torch==2.11.0' 'transformers==5.17.0' 'trl==1.7.1' \
  'peft==0.19.0' 'accelerate>=1.4,<2' 'datasets>=4.7,<5' \
  'jmespath==1.0.1' \
  'agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git@d4db4c12d52342227c4113a70183d0f64aa2f345' \
  ipykernel jupyterlab
.venv-grpo/bin/python -m ipykernel install --user \
  --name tau2-grpo-gpu --display-name 'tau2 GRPO GPU'
CUDA_VISIBLE_DEVICES=0 .venv-grpo/bin/jupyter lab
```

Select **tau2 GRPO GPU**. Check `nvidia-smi` and `torch.cuda.is_available()` if CUDA
is unavailable. If your machine needs a different CUDA build of PyTorch, install
its matching wheel using the [official PyTorch installer](https://pytorch.org/get-started/locally/)
while retaining the version contract above. Avoid a CPU-only wheel.

This is an isolated `uv pip` environment: it does not rewrite τ-bench's `uv.lock`.
AgentTune is installed from the exact Git revision in the same resolver invocation.
Install its declared dependencies: do not use `--no-deps`, because constructing
`FullClosedLoop` and the GRPO wrapper imports AgentTune's evaluator/agentic modules.
This is a larger environment than the earlier direct-TRL notebook. We explicitly
select `backend="trl"`, `PURE_TRL_MODE=1`, and `use_vllm=False`; no Unsloth/vLLM
backend is selected. Version and VCS-origin checks fail before loading model weights.

The reference's `RetrainConfig` supports DPO/BCO, so it is not used with a made-up
`algorithm="grpo"` value. Our injected `retrain_job` calls AgentTune's actual GRPO
factory and training lifecycle instead.

The first model load downloads Hugging Face weights/tokenizer unless cached.
No user-model API key is needed because every episode is solo. We explicitly use
substring communicate-info scoring, avoiding the default semantic judge's API call.
Report results as **solo + substring-scored**; do not compare them as identical to
normal conversational or semantic-scored benchmark results.

## 2. Freeze the data partitions before observing outcomes

`base` contains both training and test tasks; training on it would leak the official
test split. We first reserve validation from official `train`, then scan only the
remaining fitting tasks for failures. Official test tasks are never used for failure
selection, gradient updates, learning-rate choices, or the deployment gate.

The same split manifest is saved with the run. To resume/reproduce this exact
experiment, retain this manifest, the code revision, package versions, model revision,
configuration, and adapter. Creating a new run directory is not resuming optimizer state.

## 3. Bridge TRL's per-rollout environment to τ-bench solo Gym

TRL creates one wrapper instance per sampled completion. Within each instance,
`reset(task_id=...)` starts a **fresh** `AgentGymEnv` and returns the ticket, policy,
and complete tool catalog to the model. The initial Gym observation itself is empty.
Initialization state and evaluation criteria stay inside τ-bench; the agent sees
only the public ticket/policy/tools and tool observations.

Only one public wrapper method is exposed as a native model tool:
`execute(tool_name, arguments_json)`. It forwards a structured action to Gym.
The model can use customer-service and device-side tools because solo mode exposes
both. For example, it can request `get_customer_by_phone` with JSON phone arguments.
To end an episode it calls `execute("done", "{}")`, then emits a final response.
If it stops producing tool calls, exceeds a limit, or fails to call `done`, the
reward function finalizes the episode explicitly before reading the evaluated reward.

**Why private methods matter:** in pinned TRL 1.7.1 every public environment method
other than `reset` is exposed as a model tool. We name finish/cleanup helpers with
an underscore and supply a separate reward function. Newer docs describe a reserved
`get_reward` method, but that is not the contract used by this pinned version.

`close()` currently inherits a no-op in `AgentGymEnv`. Calling `done()` before closing
prevents unfinished Gym workers from waiting for actions. Full evaluated simulations
are saved to disk; the in-memory event list stores only small summary records.

## 4. CPU bridge smoke check before allocating GPU model memory

This intentionally performs only a customer lookup and then stops. It checks the
Gym reset/step/terminal reward path, tool catalog access, and saved trajectory.
It is **not** an easy positive training example: a lookup normally does not solve
the ticket, so a zero task reward is expected. No GPU/model is used in this cell.

## 5. Verify the pinned GPU stack

We deliberately check versions **before loading weights**. In particular, TRL's
newer environment API differs from 1.7.1, so silently upgrading it can expose helper
methods as model tools or misalign evaluation environments. Use this notebook's
isolated kernel instead of an unrelated global Python environment.

A single process must see one GPU. `device_map="auto"` is not used for training;
Accelerate manages model placement. BF16 is used where supported, otherwise FP16.
We use standard SDPA attention and no vLLM, FlashAttention build, or bitsandbytes
requirement. That reduces installation-specific dependencies for the first run.

## 6. Prepare AgentTune's native GRPO environment path

`create_agentic_trainer("grpo", backend="trl", ...)` owns the trainer lifecycle.
AgentTune forwards our `environment_factory` to its underlying TRL trainer.
We do **not** also pass `tools=` or a custom `rollout_func`: at this revision those
can activate a different AgentTune rollout path. The per-completion Gym instances,
native tool feedback, and generated-token masks stay under TRL's environment path.
AgentTune combines our single reward function with weight 1, preserving its values.

The same task grouping rules from the direct-TRL notebook apply: G generations,
one training microbatch, G accumulation/generation steps, and an evaluation batch of G.
We set warmup to zero explicitly because AgentTune's default ten warmup steps would
cover the entire small default training run.

The baseline trainer is prepared with `setup_data()` / `setup_trainer()` for evaluation
only. Later we **release it** and create a fresh AgentTune owner inside `retrain_job`.
Calling AgentTune's `train()` invokes setup again, so reusing an already attached PEFT
model would risk retaining two models or reinitializing adapters unexpectedly.
A callback captures the actual training initialization after the new trainer is built.

## 7. Collect real baseline failures from fitting tasks

`trainer.evaluate()` generates on-policy attempts but performs no optimizer update.
We run it on the fitting sample with the initial LoRA policy and save every complete
episode. An infrastructure error aborts collection; it is not converted into a zero
reward and mixed into the training set.

We retain a task if **at least one sampled baseline attempt failed** (`task_reward < 1`).
This includes partially solved tasks where the policy already has a chance of success,
which is useful for GRPO's within-group reward variance. Tasks on which every attempt
succeeded need no repair in this round. The dataset contains task IDs/prompts only;
the stored failed completions are evidence and debugging artifacts, not imitation data.

## 8. Convert actual failed episodes into FullClosedLoop buffer records

The reference calls `loop.runner.submit(TrainingExample(...))` after creating examples.
We do the same, with **negative-only failure records**: `rejected` contains serialized
actual actions from the saved failed episode, `chosen=None`, and `is_negative_only=True`.
No successful response is invented. The negative trace makes the record count as accepted
under AgentTune's buffer accounting, which accepts a non-empty chosen or rejected trace.

These are **routing/evidence records for a custom GRPO job**, not DPO training pairs.
The GRPO job uses each trajectory ID to recover its task ID from a validated failure
index, deduplicates task IDs, and samples fresh attempts. It never optimizes the
`rejected` tokens or treats the negative trace as a successful demonstration.
`root_cause="unclassified_task_failure"` is a diagnostic label, not an LLM diagnosis.

We also save an audit log of real evaluated episodes. We do not call `ingest_once()`:
its optional classifier/generator path is designed to generate corrections, and is
not required for on-policy GRPO. This direct-submit boundary matches the reference.

## 9. Check the sparse reward signal and measure baseline validation

For a task group with rewards `[0, 0, 0, 0]`, all centered advantages are zero.
GRPO cannot discover a useful repair just because the notebook ran without errors.
The strongest first run starts from a model that sometimes resolves these tasks.
If all groups tie, use a stronger starting model, a larger completion/action budget,
more exploration, or a separately validated supervised warm start. Do not invent
successful rewards or add an LLM judge to τ-bench's reward to make a curve look better.

This notebook uses **only actual terminal task reward**. It does not optimize a
lookup-success bonus, syntax bonus, hidden reference action match, or judge score.
The small default run is a hardware/integration experiment, not a convergence promise.

## 10. Inject a real GRPO retraining job into FullClosedLoop

This is the GRPO equivalent of the reference's `make_retrain_job(adapter_dir)`.
The function receives the drained buffer batch, validates its provenance against
our failure index, and constructs task-only rows for on-policy sampling. AgentTune's
`create_agentic_trainer("grpo", backend="trl", ...)` and **its own `train()`** run in
the loop's background worker. AgentTune saves weights/config/stats/provenance, and
our callback checks whether any trainable adapter tensor actually changed.

The baseline trainer will be released before firing this job. At most one GPU model
is live for generation/training at a time; the old gate runner uses cached real
baseline validation scores. Main-thread code waits while GPU training and validation
run sequentially in the worker. This is one bounded round, not a multi-GPU daemon.

## 11. Inject a held-out reward gate and real candidate runner

The reference builds gate cases from successful past pipelines. For τ-bench we use
**every fixed validation task**, including tasks the baseline fails. Selecting only
past successes would miss potential repairs. `Tau2ValidationGate` overrides only case
construction/scoring plus two explicit promotion requirements; the inherited
`evaluate_decision` and `apply_decision` still implement AgentTune's gate flow.

Old runner: cached per-task means from the real baseline validation episodes.
New runner: first verifies the saved adapter matches the in-memory trained parameters,
then executes actual solo validation episodes and returns their per-task means.
The gate compares mean terminal task reward and forbids observed regression. This
notebook additionally requires a changed adapter and **strict validation improvement**;
a tied zero score is not promoted as self-healing. A larger validation sample is
needed for a credible quality claim; this small gate is not a significance test.

The saved adapter check uses PEFT's state-dict helper so the keys match its adapter
serialization format. Neither a fake model path nor a hardcoded new-model verdict
can get through this path. Optional LLM trajectory judging is disabled and never
feeds the τ-bench reward.

## 12. Run the reference FullClosedLoop sequence

This is the same control sequence as the supplied notebook: construct the loop,
submit examples, `tick()`, wait for background retraining, inspect `loop.cycles[-1]`.
`FullClosedLoop` constructs its optional classifier/generator/evaluator objects, which
is why the complete AgentTune training environment is installed. We do not call
those LLM services; direct submission supplies the failure records and GRPO discovers
new attempts on-policy. `collect_trajectories=None` disables the optional trajectory
judge. `validation_script` is never invoked and does not stand in for task validation.

The baseline model is released before the tick. The background worker owns GPU training
and then candidate validation; the main thread only waits and prints progress. Do not
run another model/evaluation cell concurrently. Interrupting a Jupyter wait does not
cancel the daemon training thread: **restart the kernel to cancel the run** before
starting another one. The four-hour budget is configurable and is not a runtime claim.

A rejected gate is a valid completed cycle: the baseline stays selected. Missing
training artifacts, failed retraining, or a missing gate callback are errors and raise.

## 13. Final paired test evaluation after the gate is frozen

This is the first time official test tasks are run. We compare the original base
policy (LoRA disabled) and the candidate on the same frozen task sample. These
results are for reporting only and do not revise the deployment manifest or trigger
another optimization round. If you repeatedly inspect this test set and tune the
experiment from its results, it has become a development set rather than a final test.

Report raw task rewards and the exact solo/substring protocol. The dataset sample is
small, so a reported increase is an observation under this protocol, not an estimate
of benchmark-wide generalization or statistical significance.

## 14. Reload the saved adapter without loading a second model

This checks that the saved artifact can be loaded back into PEFT and that at least
one changed LoRA tensor matches the trained value. We first disable all Gym workers.
Then we save a CPU copy of one small probe tensor, remove the trainer/model from
GPU memory, load the exact base revision, and attach the saved adapter for inference.
The adapter is loaded with `is_trainable=False` intentionally: deployment is inference.

If no tensor changed, use any adapter parameter as a serialization check but keep
`weights_changed=False` in the report. Loading a zero-update artifact is not learning.
This cell does not rerun the test set or affect the gate decision.

## 15. Artifacts and how this maps to the supplied reference

| Reference component | This GPU τ-bench implementation |
|---|---|
| `make_examples()` | Actual failed solo episodes → negative-only `TrainingExample` evidence records |
| `loop.runner.submit(ex)` | Same AgentTune buffer API; no fabricated chosen response |
| `RetrainingTrigger` / `tick()` | Same real trigger and background job launch |
| `make_retrain_job(...)` | Real AgentTune GRPO factory + `owner.train()` over failed-task IDs |
| `build_model_runner(path)` | Saved-weight verification plus actual solo validation rollouts |
| Successful-audit gate cases | All preassigned validation tasks, including baseline failures |
| `DeploymentGate.evaluate_decision` | Inherited reward comparison plus changed-weight / strict-gain guards |
| `apply_decision` | Same API; atomically update a local adapter-selection manifest or keep baseline |
| Adapter reload proof | Reload base revision + saved PEFT adapter and compare a trained tensor |

Primary artifacts are under the printed run directory:

- `tau2_audit.jsonl`, `episodes.jsonl`, `episode-*.json`: actual evaluated baseline/training/
  validation/test episodes and complete action histories.
- `failed_episode_index.json`, `buffer_records.jsonl`, `drained_grpo_dataset.jsonl`:
  failure provenance, real buffer payloads, and the task IDs actually drained for GRPO.
- `candidate_adapter/`: actual LoRA weights, tokenizer, effective chat template,
  AgentTune `training_stats.json`, `grpo_training_config.yaml`, and `lexsi_provenance.json`.
- `closed_loop_cycle.json`: the actual `LoopCycle` including trigger, retrain status,
  gate decision, and `deployed` / `kept_old` result.
- `deployment_manifest.json`, `final_report.json`: frozen selection and final held-out scores.
- Configuration, package versions, exact AgentTune/model/source revisions and split
  manifest are saved separately for reproducibility.

**Sparse rewards:** tied zero-reward groups give no useful GRPO advantage. Inspect the
failed traces and consider a stronger model, more sampling/rollout budget, or a separate
validated warm start. No synthetic bonuses or judges are added to the task reward.
An adapter that does not change, or does not improve validation, is kept out of deployment.

**GPU memory:** start with the small run, use the 0.6B model / group size 2 if needed,
and restart the kernel after an OOM. Baseline and candidate are not kept simultaneously
on GPU. Do not trim policy text silently; long valid telecom rollouts can require more
memory and time than a short integration run.

**Scope:** one real training/gating cycle on one GPU. No hosted model is fine-tuned,
no production endpoint is modified, and no test sample is fed back into training.
Repeated closed-loop rounds need fresh fitting failures and a development validation
protocol; reserve final test evaluation for after model selection.

References:

- [Requested self-healing flow](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/03_self_healing_and_closed_loop.ipynb)
- [Pinned reference used here](https://github.com/Lexsi-Labs/AgentTune/blob/d4db4c12d52342227c4113a70183d0f64aa2f345/docs/notebooks/03_self_healing_and_closed_loop.ipynb)
- [AgentTune GRPO wrapper](https://github.com/Lexsi-Labs/AgentTune/blob/d4db4c12d52342227c4113a70183d0f64aa2f345/src/agenttune/backends/trl/agentic/grpo/agentic_grpo.py)
- [τ-bench Gym README](src/tau2/gym/README.md)
- [Pinned TRL environment implementation](https://github.com/huggingface/trl/blob/v1.7.1/trl/trainer/grpo_trainer.py)
