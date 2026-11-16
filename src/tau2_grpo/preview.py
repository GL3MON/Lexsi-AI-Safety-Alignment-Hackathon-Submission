"""Prompt-based tool preview for tokenizers without native tool support."""

import json
import re

import torch
from transformers import AutoModelForCausalLM, set_seed

from tau2_grpo.rollout import BASE_PROMPT, Tau2SoloRollout, set_phase


TOOL_INSTRUCTIONS = '''You control the environment through one function: execute.
For a tool call, output exactly one JSON object inside <tool_call> tags.
The outer object has keys "name" and "arguments". "arguments" is an object,
with "tool_name" (string) and "arguments_json" (a JSON-encoded string).
Example for finishing an episode:
<tool_call>
{"name":"execute","arguments":{"tool_name":"done","arguments_json":"{}"}}
</tool_call>
For any other action, replace done with a tool from the supplied Gym catalog,
and encode that tool's parameters inside arguments_json using valid JSON.
Call one tool at a time, then wait for its observation. Do not invent results.
'''


def run_prompt_preview(cfg, run, tokenizer, revision, dtype, task_id):
    """Generate real tool turns, preserving model text and dispatch errors."""
    set_phase("preview", [task_id])
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
    ).to("cuda").eval()
    for sample in range(cfg.group_size):
        set_seed(cfg.seed + sample)
        environment = Tau2SoloRollout()
        messages = []
        raw_turns = []
        try:
            context = environment.reset(task_id)
            messages = [
                {"role": "system", "content": BASE_PROMPT + "\n\n" + TOOL_INSTRUCTIONS},
                {"role": "user", "content": context},
            ]
            stem = run / f"completion-{environment._rollout_id}"
            (run / "preview-system-prompt.txt").write_text(messages[0]["content"])
            remaining = cfg.max_completion_tokens
            for turn in range(cfg.max_tool_iterations):
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to("cuda")
                available = cfg.context_limit - inputs["input_ids"].shape[1]
                if available <= 0:
                    raise ValueError(
                        f"Preview prompt needs {inputs['input_ids'].shape[1]} tokens; "
                        f"model context limit is {cfg.context_limit}. "
                        "No model output was generated for this turn."
                    )
                budget = min(remaining, available, 512)
                if budget <= 0:
                    break
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=budget, do_sample=True,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                ids = output[0, inputs["input_ids"].shape[1]:]
                remaining -= len(ids)
                text = tokenizer.decode(ids, skip_special_tokens=True)
                raw_turns.append(tokenizer.decode(ids, skip_special_tokens=False))
                messages.append({"role": "assistant", "content": text})
                print(f"Model output ({sample + 1}, turn {turn + 1}):\n{text}", flush=True)
                match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
                if match is None:
                    break
                try:
                    call = json.loads(match.group(1))
                    if call["name"] != "execute":
                        raise ValueError("The available function is execute")
                    observation = environment.execute(**call["arguments"])
                except (ValueError, TypeError, KeyError) as error:
                    observation = json.dumps({"error": str(error)})
                messages.append({"role": "user", "content": "<tool_response>\n" + observation + "\n</tool_response>"})
                # Persist after each turn so partial results remain inspectable.
                stem.with_suffix(".json").write_text(json.dumps({
                    "phase": "preview", "task_id": task_id,
                    "messages": messages, "raw_model_turns": raw_turns,
                }, indent=2))
                stem.with_suffix(".txt").write_text("\n\n".join(
                    m["role"].upper() + ":\n" + m["content"] for m in messages
                ))
                if environment._gym_done or remaining <= 0:
                    break
            # Also save plain-text responses that contained no tool call.
            stem.with_suffix(".json").write_text(json.dumps({
                "phase": "preview", "task_id": task_id,
                "messages": messages, "raw_model_turns": raw_turns,
            }, indent=2))
            stem.with_suffix(".txt").write_text("\n\n".join(
                m["role"].upper() + ":\n" + m["content"] for m in messages
            ))
            print(f"Saved model output: {stem.with_suffix('.txt')}", flush=True)
        finally:
            if environment._env is not None:
                environment._finish()
