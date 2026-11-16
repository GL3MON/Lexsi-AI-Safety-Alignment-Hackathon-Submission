"""Local retail conversation preview with direct tool calls and a local user."""

import json
import random
import re
import sys
from dataclasses import asdict
from uuid import uuid4


def parse_call(text):
    """Parse a single explicit tool call; plain text is a customer response."""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if match is None:
        if "<tool_call>" in text:
            raise ValueError("Incomplete tool call; close the </tool_call> tag")
        return None
    call = json.loads(match.group(1))
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError("Tool call needs a string name")
    if not isinstance(call.get("arguments"), dict):
        raise ValueError("Tool call arguments must be a JSON object")
    return call


def run_retail_preview(cfg, output):
    import torch
    from loguru import logger
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

    from tau2.data_model.message import UserMessage
    from tau2.data_model.semantic_communication import CommunicateScorer, CommunicateScoring
    from tau2.domains.retail.environment import get_tasks_split
    from tau2.gym.gym_agent import AgentGymEnv
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    run = output / f"run-{uuid4().hex[:10]}"
    run.mkdir(parents=True)
    print(f"Artifacts: {run}", flush=True)
    (run / "experiment_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    base = AutoConfig.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    revision = base._commit_hash or cfg.model_revision
    if cfg.context_limit > base.max_position_embeddings:
        raise ValueError("Configured context exceeds model context")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, revision=revision, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    def generate(messages, max_tokens):
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        count = inputs["input_ids"].shape[1]
        available = cfg.context_limit - count
        if available <= 0:
            raise ValueError(f"Prompt has {count} tokens; limit is {cfg.context_limit}")
        with torch.inference_mode():
            tokens = model.generate(
                **inputs, max_new_tokens=min(max_tokens, available),
                do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )[0, count:]
        return (
            tokenizer.decode(tokens, skip_special_tokens=True),
            tokenizer.decode(tokens, skip_special_tokens=False), len(tokens), count,
        )

    simulator_log = []

    class LocalUser(UserSimulator):
        def generate_next_message(self, message, state):
            if message.has_content():
                state.messages.append(message)
            history = state.system_messages + state.flip_roles()
            messages = [{"role": m.role, "content": m.content or ""} for m in history]
            if len(messages) == 1:
                messages.append({"role": "user", "content": "Begin the customer conversation."})
            text, raw, _, count = generate(messages, 256)
            simulator_log.append({"messages": messages, "raw_output": raw, "prompt_tokens": count})
            response = UserMessage(role="user", content=text)
            state.messages.append(response)
            print(f"CUSTOMER: {text}", flush=True)
            return response, state

    registry.register_user(LocalUser, "grpo_preview_local_user")

    class LocalRetailGym(AgentGymEnv):
        def _run_config(self, seed=None):
            return super()._run_config(seed).model_copy(update={"user": "grpo_preview_local_user"})

    ids = sorted(get_tasks_split()["train"])
    random.Random(cfg.seed).shuffle(ids)
    task_id = ids[0]
    (run / "runtime.json").write_text(json.dumps({
        "model_id": cfg.model_id, "revision": revision, "task_id": task_id,
        "domain": "retail", "mode": "preview", "user_model": cfg.model_id,
        "scoring": "substring", "training": False,
    }, indent=2))
    for sample in range(cfg.group_size):
        set_seed(cfg.seed + sample)
        simulator_log.clear()
        env = LocalRetailGym(
            domain="retail", task_id=task_id, solo_mode=False,
            user_llm="local-preview", user_llm_args={},
            max_steps=2 * cfg.max_actions + 4,
            communicate_scoring=CommunicateScoring(scorer=CommunicateScorer.SUBSTRING),
        )
        messages, turns = [], []
        stem = run / f"completion-{sample + 1}"

        def save():
            stem.with_suffix(".json").write_text(json.dumps({
                "task_id": task_id, "messages": messages, "model_turns": turns,
            }, indent=2))
            stem.with_suffix(".txt").write_text("\n\n".join(
                m["role"].upper() + ":\n" + m["content"] for m in messages
            ))
            (run / f"user-simulator-{sample + 1}.json").write_text(json.dumps(simulator_log, indent=2))

        try:
            observation, info = env.reset(seed=cfg.seed + sample)
            if not observation:
                raise RuntimeError("Retail user simulator did not produce an opening message")
            schemas = [tool.openai_schema for tool in info["tools"]]
            names = {tool.name for tool in info["tools"]}
            system = (
                "You are a retail customer support agent. Follow the policy below. "
                "Use only the available tools and actual observations. "
                "For a tool call, output exactly one JSON object inside <tool_call> tags. "
                "The name is a tool name; arguments is an object, not a string.\n"
                'Example: <tool_call>{"name":"get_user_details","arguments":{"user_id":"example_user"}}</tool_call>\n'
                "Replace example values with real values obtained from the customer or tools. "
                "Wait for each tool response. To talk to the customer, output plain text "
                "without a tool call. Finish with the done tool only after helping the customer.\n\n"
                + info["policy"] + "\n\nAvailable tools:\n" + json.dumps(schemas)
            )
            messages.extend([{"role": "system", "content": system}, {"role": "user", "content": observation}])
            (run / "system-prompt.txt").write_text(system)
            remaining = cfg.max_completion_tokens
            for turn in range(cfg.max_actions):
                text, raw, used, count = generate(messages, min(512, remaining))
                remaining -= used
                print(f"AGENT ({sample + 1}, turn {turn + 1}, prompt {count} tokens): {text}", flush=True)
                turns.append({"raw_output": raw, "prompt_tokens": count, "generated_tokens": used})
                messages.append({"role": "assistant", "content": text})
                save()
                try:
                    call = parse_call(text)
                    if call is not None and call["name"] not in names:
                        raise ValueError(f"Unknown tool: {call['name']}")
                except (ValueError, TypeError) as error:
                    messages.append({"role": "user", "content": "Tool format error: " + str(error)})
                    save()
                    if remaining <= 0:
                        break
                    continue
                action = json.dumps(call) if call else text
                observation, reward, terminated, truncated, info = env.step(action)
                messages.append({"role": "user", "content": observation or "Episode finished."})
                save()
                if terminated or truncated or remaining <= 0:
                    break
            if not env._simulation_done.is_set():
                _, _, _, _, info = env.step('done()')
            simulation = json.loads(info["simulation_run"])
            if not simulation.get("reward_info"):
                raise RuntimeError("Retail simulation ended without an evaluated reward")
            (run / f"episode-{sample + 1}.json").write_text(json.dumps(simulation, indent=2))
            print(f"Saved: {stem.with_suffix('.txt')}; reward: {simulation['reward_info']['reward']}", flush=True)
        finally:
            save()
            if env._agent is not None and not env._simulation_done.is_set():
                env.step('done()')
            env.close()
    print("Retail preview complete.", flush=True)
