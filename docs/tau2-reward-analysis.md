# Tau2 reward distribution

This runs inference against real Tau2 environments and records their terminal task rewards. No optimizer or training is run. Retail uses a user simulator; by default the same endpoint/model plays the user and agent. Telecom runs solo. Communication checks use Tau2's substring scorer. These choices are saved in settings.json.

## Serve on MIG2

The separate environment preserves the SFT environment:

```bash
uv venv .venv-vllm --python .venv/bin/python
uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt
scripts/serve_sft_mig2.sh
```

The launcher applies the local vLLM 0.8.5 MIG UUID patch and creates a hardlinked compatibility view of checkpoint 525's merged export. The view restores the older names for RoPE theta and interleaved sliding windows, and the older tokenizer class name. Original exported files stay available. The patch resolves MIG UUIDs via NVML and queries parent GPU compute capability while keeping partition-specific UUID and memory queries. A backup of the installed source and a unified patch are saved. Torch compilation is disabled for the older Triton/Python combination. A second package patch replaces the compiled Cohere normalization helper with native fused LayerNorm; an FP32/BF16 numerical comparison passed (sampled BF16 maximum absolute difference 0.0078125). CUDA graph capture remains enabled for decode. The same 8-token prompt took 24.8 seconds in eager mode and 3.2 seconds with graphs, including prefill. Serving uses CUDA graphs for decode, BF16 weights and compute, chunked prefill with 512 tokens per batch, prefix caching, one concurrent sequence, an 8192-token context, and 36% of the MIG partition's memory budget. The measured profile used 6.28 GiB for weights, 0.07 GiB peak activations, and 0.62 GiB for KV cache. Change settings in the launcher or override CLI flags. The earlier 4-bit attempt decoded at only about 0.3 tokens/sec and was interrupted before completing an episode.

## Analyze training tasks

```bash
.venv/bin/python scripts/analyze_tau2_rewards.py \
  --domain retail --tasks 8 --samples-per-task 4
```

A small initial run:

```bash
.venv/bin/python scripts/analyze_tau2_rewards.py \
  --domain retail --tasks 2 --samples-per-task 2 --max-actions 16
```

Tasks are selected by a deterministic shuffle of the official training split. Each episode starts from a fresh environment. The model receives policy, task observations, and tool schemas, without reference actions or reward criteria. Model text is parsed for direct `<tool_call>` blocks, schema validated, and executed against the environment. All failures and rejected tool calls are retained in the transcript. An unfinished episode is externally ended after the action budget and its actual terminal reward is recorded; `forced_stop` identifies these episodes. Missing evaluations and server errors produce null rewards, not zeros, and are excluded from the reward histogram. Neither policy nor history is silently shortened to fit context.

Output under `outputs/tau2-reward-analysis/run-*`:

- `settings.json`: selected task IDs, seeds, model endpoint and simulator choice.
- `episode-NNNN.json` and `.txt`: prompts, raw model completions, rejected calls, tool observations, simulator generations, and full Tau2 simulation.
- `rewards.jsonl`: one record per episode, including actual reward details or error.
- `summary.json`: empirical reward counts, mean, standard deviation, success fraction and per-task variation.
- `reward_distribution.png`: saved distribution plot.

W&B logging is online by default in `tiny-aya-function-calling`; `--wandb-mode offline` is available. Increase task/sample counts for a useful distribution; a four-episode run only checks the inference and reward pipeline. Scores with a small locally simulated user should not be treated as comparable to a benchmark using a stronger external user model.

A Transformers fallback server is also available at `scripts/serve_sft_transformers.py`. Its unquantized BF16 mode needs more free memory than the shared MIG2 slice had during the first attempt.

## Structured observations

The v2 bridge reads `env._agent.observation` and selects incoming messages after the latest assistant action. It preserves customer text as `role: user`, saves tool results as `role: tool` with their names and call IDs, and serializes result content as `{"name": ..., "content": ...}`. The Aya Jinja template maps tool messages to user-turn tokens with explicit `<tool_response>` wrappers, matching the SFT representation. Runtime validation errors are tagged as runtime tool feedback. Gym display strings such as `tool: ...` are not used as model inputs. Each model turn includes its exact `rendered_prompt` for inspection.

`run-339f994955` used the earlier flattened observation bridge; its `analysis_validity.json` marks it unsuitable for assessing the model. Its original transcripts and measured rewards remain preserved.
