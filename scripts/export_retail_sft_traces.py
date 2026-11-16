"""Export successful Tau2 conversations into retail SFT candidates for policy review."""

import argparse
import json
from pathlib import Path

from jsonschema import ValidationError, validate

from tau2_grpo.prepare_function_calling import model_messages, parse_calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/retail-data/split_manifest.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-all-rewards", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; use a new path")
    manifest = json.loads(args.manifest.read_text())
    settings = json.loads((args.run / "settings.json").read_text())
    rows = []
    rejected = []
    for path in sorted(args.run.glob("episode-*.json")):
        try:
            episode = json.loads(path.read_text())
            record = episode["record"]
            task_id = str(record["task_id"])
            partition = next(
                (
                    k
                    for k in ("sft", "validation")
                    if task_id in manifest["partitions"][k]
                ),
                None,
            )
            if partition is None:
                raise ValueError("reserved_non_sft_task")
            if record.get("error") or record.get("reward") is None:
                raise ValueError("no_evaluated_episode")
            if not args.include_all_rewards and record.get("reward") != 1.0:
                raise ValueError("no_successful_reward")
            if record.get("forced_stop") and not args.include_all_rewards:
                raise ValueError("unfinished_conversation")
            tools = record["tools"]
            schemas = {
                t["function"]["name"]: t["function"]["parameters"] for t in tools
            }
            messages = []
            for m in episode["messages"]:
                if m.get("origin") == "validation":
                    raise ValueError("rejected_model_action")
                content = m.get("content") or ""
                if m["role"] == "assistant" and m.get("tool_calls"):
                    content = "\n".join(
                        "<tool_call>"
                        + json.dumps(
                            {
                                "name": c["function"]["name"],
                                "arguments": json.loads(c["function"]["arguments"])
                                if isinstance(c["function"]["arguments"], str)
                                else c["function"]["arguments"],
                            }
                        )
                        + "</tool_call>"
                        for c in m["tool_calls"]
                    )
                if m["role"] == "assistant":
                    for call in parse_calls(content):
                        if call["name"] not in schemas:
                            raise ValueError("unknown_tool")
                        validate(call["arguments"], schemas[call["name"]])
                messages.append(
                    {
                        k: v
                        for k, v in dict(m, content=content).items()
                        if k in {"role", "content", "name"}
                    }
                )
            messages = model_messages(messages)
            messages[0]["content"] += "\n\n<tools>\n" + json.dumps(tools) + "\n</tools>"
            rows.append(
                {
                    "task_id": task_id,
                    "partition": partition,
                    "messages": messages,
                    "tools": tools,
                    "source_episode": str(path.resolve()),
                    "teacher_model": settings["model"],
                    "user_model": settings["user_model"],
                    "customer_profile": settings.get("customer_profile", "standard"),
                    "agent_prompt_sha256": settings.get("agent_prompt_sha256"),
                    "user_prompt_sha256": settings.get("user_prompt_sha256"),
                    "reward": record["reward"],
                    "source_forced_stop": bool(record.get("forced_stop")),
                    "policy_review": "disabled"
                    if args.include_all_rewards
                    else "pending",
                    "model_settings": {
                        "api_mode": settings["api_mode"],
                        "temperature": settings["temperature"],
                        "user_temperature": settings.get(
                            "user_temperature", settings["temperature"]
                        ),
                        "context_limit": settings["context_limit"],
                        "server_profile": settings.get("server_profile"),
                    },
                }
            )
        except (ValueError, KeyError, ValidationError) as error:
            rejected.append({"episode": str(path), "reason": str(error)})
    args.output.mkdir(parents=True)
    for partition in ("sft", "validation"):
        subset = [r for r in rows if r["partition"] == partition]
        if subset:
            (args.output / (partition + "-candidates.jsonl")).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset)
            )
    report = {
        "candidate_conversations": len(rows),
        "rejected": rejected,
        "requires_policy_review": not args.include_all_rewards,
        "reward_filter": "none" if args.include_all_rewards else "reward_1",
        "note": "Reward is retained as metadata; structural checks still apply."
        if args.include_all_rewards
        else "Candidates require preparation before training.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
