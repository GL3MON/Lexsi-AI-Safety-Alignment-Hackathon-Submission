# Deferred: context windows for retail RL

Current status: the retail harness is implemented; see [retail RL](retail-rl.md). SFT has completed. The historical planning notes below describe the starting state. No full RL optimization run has been launched.

Original status: deferred at the user's request on 2026-11-09. Finish retail SFT first. No RL training or context-window implementation was started; the last RL work was inspection of the existing code and installed trainer APIs.

## Resume here after SFT

Implement context management before each agent action while preserving one continuous Tau2 environment episode. A context-window boundary must not reset the environment or create a new reward-bearing episode.

- Student total context: 8,192 tokens. Reserve 512 tokens for the next response, leaving a maximum prompt of 7,680 tokens, including the chat template, policy and tool schemas.
- Preserve the system policy/tool catalog. Remove older complete exchanges when the prompt exceeds its budget; never cut tool-call JSON or leave an orphan observation.
- Use a shared, causal history-selection rule for RL rollout generation, training replay, SFT preparation and inference. Selection may only use history available before the next action, never a reference action, future tool result or final reward.
- Record the exact rendered prompt and prompt token IDs used for every action, generated token IDs, behavior-policy version and aligned old-policy token log probabilities. Replay those exact contexts when computing new-policy and reference-policy probabilities.
- Keep the complete raw episode separately. Removing an action from a later prompt must not remove its original training record.
- Assign the actual Tau2 terminal reward once per episode. Compute GRPO group advantages across independent episodes of the same task, then associate the episode's advantage with its generated action tokens. Repeated context, tool outputs and customer messages receive no policy loss. Avoid counting one trajectory as several independent reward samples simply because it has multiple context windows.
- Keep policy weights fixed during a rollout group. Preserve correct episode/token normalization across trajectories of different lengths. Infrastructure failures must not become fabricated reward-zero labels.

## Existing SFT prototype and its limits

`scripts/split_retail_sft_traces.py` exports overlapping conversation chunks from the full saved candidates. It repeats policy/tools and preserves complete tool exchanges. Its `loss_message_indices` metadata is honored by both current SFT backends to avoid duplicate assistant supervision. It does not change the active collection or SFT configuration.

The 11K demonstration is in `outputs/retail-data/split-11k-example/`: three chunks of 7,351, 8,165 and 7,771 tokens, covering all 20 assistant turns once. Two chunks lose original customer/tool sources for some later arguments. The report flags these gaps; some values remain visible in assistant explanations. Splitting alone is not evidence-preserving context management and the report does not establish policy compliance. These training chunks also use the full 8K budget; an online RL prompt must additionally leave room for its next output.

Before deployment, resolve how to retain authentication, active order status, original/new item IDs and options, payment information and explicit confirmation. A fixed overlap cannot guarantee that these facts remain available. If policy plus required evidence cannot fit, fail clearly or use a deliberately designed state-compression approach; do not silently invent missing state.

## Integration points found during inspection

- `src/tau2_grpo/rollout.py`: existing solo-mode Gym bridge and terminal reward persistence. It currently uses an `execute` wrapper and builds a full ticket/policy/catalog prompt.
- `src/tau2_grpo/experiment.py`: existing AgentTune-owned TRL GRPO setup. It currently asserts native environment rollouts and has no custom rollout function.
- `src/tau2_grpo/cli.py`: retail currently supports preview, not RL training. A real retail RL entry point still needs wiring; changing only a history helper is insufficient.
- `src/tau2_grpo/config.py`: existing experiment budgets describe the legacy full-trajectory path. Keep per-response generation reserve distinct from the episode action/token budget.
- `scripts/analyze_tau2_rewards.py`: existing interactive retail inference bridge, structured tool observations, local customer simulator and local NL reward judge. These are separate from the legacy solo GRPO path.

Pinned TRL 1.7.1's native tool loop accumulates tool outputs and model tokens into one completion. Its normal loss path reconstructs one prompt plus that completion. Windowed per-action prompts cannot simply be flattened into that sequence: it would recompute probabilities under the wrong history. Implement and verify an explicit turn-aware rollout/training path, preserving episode-level grouping. Current GRPO `rollout_func` supports prompt IDs, completion IDs and log probabilities, but its single-row-per-rollout contract is not automatically a solution for variable numbers of turn windows.

## Checks required before a training run

1. Force several context-window shifts in one fake episode; assert the environment resets only once and scores once.
2. Verify every model request leaves its output reserve, retains policy/tools and has intact tool exchanges.
3. Assert training uses the exact saved generation token IDs and behavior-policy log probabilities after every shift.
4. Check loss masks, GRPO group alignment, episode weighting and one-time coverage of generated action tokens; exclude observations and repeated context.
5. Compare rollout-time and replay-time log probabilities before any optimizer update using a small local model; then verify gradients with a CPU smoke test.
6. Run a small real retail smoke test after SFT and after explicitly allocating GPU capacity. Do not interrupt the current trace collectors or teacher servers for this deferred work.

References: [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer), [TRL environment integration](https://huggingface.co/docs/trl/openenv), [fixed-context overlapping windows](https://huggingface.co/docs/transformers/perplexity).

## Immediate SFT priority

Continue the existing MIG2/MIG0 Qwen collectors: three generations per SFT task and one per validation task, without a model reviewer or success-quota retries. Combined selected files are under `outputs/retail-data/qwen-sft-parallel/`; raw traces remain in each worker's `runs/` directory. Current SFT config still reads the combined, prefix-shortened files. Resolve the chunk/context preparation choice before launching SFT; the splitter candidates have not replaced those files. Do not infer authorization to start training from this reminder.
