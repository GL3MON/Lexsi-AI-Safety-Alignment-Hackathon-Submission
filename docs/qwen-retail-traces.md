# Qwen3.5 retail reward pilot and SFT trace candidates

Teacher and customer use `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in separate conversation contexts. The student SFT configuration is `configs/sft-retail.json`.

## Serving

```bash
uv venv .venv-vllm-qwen --python .venv/bin/python
uv pip install --python .venv-vllm-qwen/bin/python -r requirements-vllm-qwen.txt
scripts/serve_qwen_retail_mig2.sh
```

The launcher serves `qwen3.5-4b` at `http://127.0.0.1:8004/v1` on MIG2. It uses vLLM 0.17.1, original BF16 text weights, a 16384-token context and one concurrent sequence. Half the decoder layers use the built-in prefetch CPU offloader because the shared MIG2 has only about 7.5 GiB free. The initial measured model/device-buffer footprint was approximately 4.68 GiB. GPU memory utilization is capped at 0.30, with decode-only CUDA graphs and 528-token prefill batches to fit the model's hybrid cache alignment.

The launcher reapplies two local vLLM package patches, each with source backups and unified diffs:

- MIG UUID-aware NVML lookup, preserving partition-specific UUID and memory and using parent compute capability.
- Optional selection of vLLM's bundled Triton/FLA GDN prefill implementation, via `TAU2_QWEN_GDN_TRITON=1`, because the default FlashInfer prefill path requires an nvcc installation absent on this host.

The isolated environment keeps the earlier Aya environments available. No system NVIDIA driver is installed or changed by these scripts.

The same launcher accepts `QWEN_QUANTIZATION` (default `none`), `QWEN_OFFLOAD` (`prefetch` or `none`) and `QWEN_GPU_MEMORY_UTILIZATION` (default `0.30`). A full-GPU FP8 trial on MIG0 uses `QWEN_QUANTIZATION=fp8 QWEN_OFFLOAD=none QWEN_GPU_MEMORY_UTILIZATION=0.40`, with the MIG0 UUID and port 8005. This changes weights/linear computation precision, while leaving the model revision, BF16 base dtype and default KV-cache dtype unchanged. It requires actual memory and quality checks; available capacity is shared with other jobs.

On 2026-11-10, both active servers passed the four matched native-call, selection, long-context and real exchange replay probes in full-GPU FP8 mode. MIG0 uses utilization `0.40`; MIG2 uses `0.33`, retaining a 16,384-token context without CPU offloading. On MIG2 the probe total fell from 34.45 s to 5.80 s; these timings describe the probe set, not a guaranteed full-episode speedup or comprehensive quality equivalence. Reports are under `outputs/logs/mig2-fp8-benchmark/` and `outputs/logs/mig0-fp8-benchmark/strict-identifiers/`. The switch helper accepts `--worker-index 0` for MIG2 and `--worker-index 1` (default) for MIG0. The recovery supervisor reads each worker's saved profile so restarts preserve its validated mode.

The launcher records `outputs/logs/serving-PORT.json`. Newly started collector episodes snapshot this profile into their settings, and exported rows include it under `model_settings.server_profile`. The same endpoint supplies teacher, customer and local judge, so a precision change affects all three. Older rows without this profile retain their original metadata.

For a planned server change, the supervisor supports adopting existing collectors via `--adopt-worker INDEX:PID`. `qwen-sft-parallel/worker-INDEX.maintenance` prevents that worker from restarting while its collector finishes under `PAUSE_AFTER_EPISODE`. The other collector can continue. Server recovery settings persist in `server-profiles.json`. `scripts/switch_mig0_fp8.py` waits for the marked MIG0 collector to finish, benchmarks idle BF16, tests full-GPU FP8, and restores original BF16/offloading if startup, smoke checks or the latency comparison fail. Results are saved under `outputs/logs/mig0-fp8-benchmark/`, with operation progress in `outputs/logs/mig0-fp8-switch.log`. These narrow probes do not establish unchanged full retail reasoning quality. Remove the pause/maintenance markers only after a healthy server is available; the switch script does this automatically after a successful trial or rollback.

## Parallel collection

Start an additional server on MIG0:

```bash
CUDA_VISIBLE_DEVICES=MIG-efa8ccfd-2143-54d5-baf1-8f664020ef04 PORT=8005 scripts/serve_qwen_retail_mig2.sh
```

With servers on ports 8004 (MIG2) and 8005 (MIG0), run `.venv/bin/python scripts/run_parallel_retail_sft.py`. When migrating a running collector, create `outputs/retail-data/qwen-sft-fixed/PAUSE_AFTER_EPISODE` and pass `--wait-collector-pid PID` so its current episode finishes first. The supervisor starts disjoint task workers, resumes existing samples, and atomically publishes shuffled combined selections to `outputs/retail-data/qwen-sft-parallel/{sft,validation}.jsonl`. Each worker logs online to W&B. Raw conversations remain in `qwen-sft-fixed/runs` (MIG2) and `qwen-sft-fixed-mig0/runs` (MIG0). Worker assignments are saved in `qwen-sft-parallel/workers.json`; combined counts are in `progress.json`. The SFT config reads the combined files. Model policy review stays disabled.

The parallel supervisor defaults to `--collection-mode all-generations`: three completed generations per SFT task and one per validation task. It uses the first completed generations already collected, includes evaluated reward-zero episodes, and retains original rewards and stop flags as metadata. There are no retries to replace reward-zero generations and no grounding/authentication acceptance gate in this mode. Infrastructure failures are preserved separately and do not consume the generation quota; resume reconstructs counts from actual evaluated episodes. Structurally unexportable conversations are recorded rather than silently replaced. Student context preparation still applies, and a shortened example is marked `trajectory_prefix`. Switching from the previous success-quota mode archives previous selected files and state under `selection-before-mode-change-*` and rebuilds the selections. Existing extra raw attempts are preserved but are outside the first-three selection. Use `--collection-mode successful` to restore success-quota collection explicitly.

For unattended collection, run `.venv/bin/python scripts/run_parallel_retail_sft.py --manage-servers`. It restores absent local API processes on their assigned MIGs, waits for readiness, and retries failed workers up to `--max-worker-restarts` (default 3). A worker failure does not interrupt the other worker's active episode. Existing API processes warming up are left alone. Recovery state is saved in `qwen-sft-parallel/worker-status.json`; exhausted workers produce `incomplete.json` after surviving workers finish. `completed.json` is written only when both generation budgets are fulfilled. Model-generation failures and server recovery are separate from task reward filtering.

## Reward pilot

```bash
scripts/run_qwen_retail_rewards.sh
```

This runs two samples each on retail tasks 88 and 2, both reserved for SFT by the split manifest. It uses native OpenAI chat tool calls, `qwen3_coder` tool parsing, `qwen3` reasoning parsing, and `enable_thinking=false`. The agent gets policy and native schemas; the customer gets only its scenario and user conversation. Reference actions and reward checks are not supplied to the agent.

W&B project: `tau2-retail`. Each run saves readable conversations, native API messages, rendered prompts, simulator conversations, actual Tau2 terminal rewards, infrastructure errors, a summary and a reward histogram. Tool responses retain `role: tool` and their matching API call IDs. The action budget is 24 and each response has a 512-token cap. These settings and the different customer model mean the pilot is not an isolated agent-only comparison with earlier Aya runs.

## SFT trace candidates

To make a snapshot of overlapping context chunks from collected full trajectories:

```bash
.venv/bin/python scripts/split_retail_sft_traces.py \
  --input outputs/retail-data/qwen-sft-parallel/sft.jsonl \
  --output outputs/retail-data/qwen-sft-chunks \
  --max-length 8192 --overlap-tokens 1024
```

This writes `chunks.jsonl` and `report.json` to a new output directory. It reads full conversations from the collector's saved candidates, rather than splitting the already shortened selected messages. Tool calls and following observations stay together, the policy/catalog repeats, and the original initial request is repeated where needed to preserve Aya's alternating chat roles. Chunks carry `loss_message_indices`; both training backends honor these masks so repeated assistant history is context only. All chunks from a trajectory retain its task-based group ID. The active collection and training config are not changed. Export validation separately with its own input/output directory to preserve the task split.

This is a splitting prototype, not automatic evidence-preserving context management. A fixed overlap can exclude older customer/tool sources for later arguments. The report lists these as `missing_argument_evidence`; it does not verify authentication or confirmation prerequisites. Some values may still appear in retained assistant text despite missing their original tool source. Check those candidates before training and use compatible history management at inference. If a single complete exchange plus policy/catalog exceeds the context budget, the script fails rather than cutting tool JSON. More than two chunks may be required.

After a reward run completes:

```bash
.venv/bin/python scripts/export_retail_sft_traces.py \
  --run outputs/tau2-reward-analysis/run-REPLACE \
  --output outputs/retail-data/qwen-candidates-REPLACE
```

The exporter accepts naturally completed episodes with reward 1, validates tool calls and their schemas, enforces the task reservations, and converts native function calls/tool observations to Tiny Aya's supervised format. It rejects unsuccessful, externally stopped, malformed or wrongly partitioned episodes. The collector adds grounding, authentication and 8192-token preparation checks. Model policy review is disabled by default at the user's request; `--policy-review` enables it. Resuming without review retries previous reviewer rejections and preserves their audit files. Accepted examples do not claim comprehensive policy compliance.

For broader collection on the SFT task partition, the same runner supports:

```bash
scripts/run_qwen_retail_rewards.sh --task-ids '' --tasks 40 --samples-per-task 3
```

For separate validation conversations, use `--partition validation --task-ids '' --tasks 10 --samples-per-task 1`. Keep these outside training. RL and official test task IDs are rejected by the SFT trace exporter.
