from tau2_grpo.retail_memory import summary_errors, controller_history
"""Snapshot completed traces and prepare bounded, summary-conditioned SFT segments."""

import argparse
import copy
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

from tau2_grpo.prepare_function_calling import digest, parse_calls
from tau2_grpo.sft_data import tokenize_assistant_only

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def canonical(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        for tag in ("tools", "tool_call", "tool_response"):
            message["content"] = re.sub(
                "<" + tag + r">\s*(.*?)\s*</" + tag + ">",
                lambda m, tag=tag: (
                    "<"
                    + tag
                    + ">"
                    + json.dumps(
                        json.loads(m.group(1)),
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "</"
                    + tag
                    + ">"
                ),
                message["content"],
                flags=re.DOTALL,
            )
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def wire(messages):
    return [
        {
            key: value
            for key, value in m.items()
            if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
        }
        for m in messages
    ]


def values(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from values(v, k)
    elif isinstance(value, list):
        for item in value:
            yield from values(item, key)
    elif (
        key.endswith(("_id", "_ids"))
        or key
        in {
            "email",
            "zip",
            "first_name",
            "last_name",
        }
    ) and value not in (None, "", "original_payment"):
        yield str(value)


def contains(value, text):
    return bool(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
    )


def observed(messages):
    return "\n".join(m["content"] for m in messages if m["role"] in {"user", "tool"})


def lost_arguments(original, compact, target):
    before, after = observed(original), observed(compact)
    return [
        value
        for call in parse_calls(target["content"])
        for value in values(call["arguments"])
        if contains(value, before) and not contains(value, after)
    ]






def prepare(record, folder, tokenizer, client, prompt):
    source = Path(record["source_episode"])
    episode = json.loads(source.read_text())
    candidate = (
        source.parents[2]
        / "exports-all-generations"
        / source.parent.name
        / (record["partition"] + "-candidates.jsonl")
    )
    full = next(r for r in read_rows(candidate) if r["source_episode"] == str(source))
    messages = canonical(full["messages"])
    raw = wire(episode["messages"])
    if len(raw) != len(messages):
        raise ValueError("Raw and canonical message indices differ")
    count = lambda ms, generation=False: len(
        tokenizer.apply_chat_template(
            ms, tokenize=True, return_dict=False, add_generation_prompt=generation
        )
    )
    assistant_ids = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    total = count(messages)
    prepared = []
    covered = []
    audits = []

    def add(ms, source_indices, targets, memory_path=None):
        target_positions = [j for j, i in enumerate(source_indices) if i in targets]
        encoded, _ = tokenize_assistant_only(tokenizer, ms, 8192, target_positions)
        row = {
            k: v
            for k, v in record.items()
            if k not in {"messages", "id", "quality_checks"}
        }
        row.update(
            messages=ms,
            id=digest(
                {"source": str(source), "indices": source_indices, "targets": targets}
            ),
            loss_message_indices=target_positions,
            source_message_indices=source_indices,
            source_target_indices=targets,
            example_type="summary_conditioned_segment"
            if memory_path
            else "intact_conversation"
            if len(ms) == len(messages)
            else "initial_conversation_segment",
            student_tokens=len(encoded["input_ids"]),
            supervised_tokens=sum(t != -100 for t in encoded["labels"]),
            supervised_assistant_turns=len(targets),
            supervised_tool_calls=sum(
                len(parse_calls(messages[i]["content"])) for i in targets
            ),
            source_tokens=total,
            source_assistant_turns=len(assistant_ids),
            memory_path=memory_path,
            quality_checks=[
                "context_8192",
                "original_targets_unchanged",
                "assistant_only_loss",
                "no_duplicate_target_coverage",
            ],
        )
        prepared.append(row)
        covered.extend(targets)

    if total <= 8192:
        add(messages, list(range(len(messages))), assistant_ids)
    else:
        ends = [i for i in assistant_ids if count(messages[: i + 1]) <= 8192]
        if not ends:
            raise ValueError("Policy and first assistant response exceed 8K")
        end = ends[-1]
        add(
            messages[: end + 1],
            list(range(end + 1)),
            [i for i in assistant_ids if i <= end],
        )
        remaining = [i for i in assistant_ids if i > end]
        while remaining:
            target = remaining[0]
            previous = max(i for i in assistant_ids if i < target)
            tail = copy.deepcopy(messages[previous:target])
            prefix = raw[:target]
            numbered = [{"message_id": i, **m} for i, m in enumerate(prefix) if i]
            payload = {
                "policy": raw[0]["content"],
                "original_request": raw[1]["content"],
                "history_to_summarize": numbered[: previous - 1],
                "recent_messages_kept_verbatim": numbered[previous - 1 :],
                "recent_tail_message_ids": list(range(previous, target)),
            }
            memory = None
            compact = None
            attempt_errors = []
            for attempt in range(3):
                request = [
                    {
                        "role": "system",
                        "content": prompt
                        + (
                            "\nRetry: retain the current requests' verified identifiers, profile/payment information and constraints; use fewer words."
                            if attempt
                            else ""
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
                attempt_file = (
                    folder / f"summary-before-{target}-attempt-{attempt}.json"
                )
                cached = (
                    json.loads(attempt_file.read_text())
                    if attempt_file.exists()
                    else {}
                )
                if cached.get("input") == request:
                    response = cached["response"]
                else:
                    response = client.chat.completions.create(
                        model="qwen3.5-4b",
                        messages=request,
                        temperature=0 if attempt == 0 else 0.2,
                        seed=2026 + attempt,
                        max_tokens=2000,
                        response_format={"type": "json_object"},
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    ).model_dump()
                    atomic(attempt_file, {"input": request, "response": response})
                if response["choices"][0]["finish_reason"] == "length":
                    attempt_errors = ["truncated_summary"]
                    continue
                try:
                    candidate_memory = json.loads(
                        response["choices"][0]["message"]["content"] or ""
                    )
                except ValueError:
                    attempt_errors = ["invalid_summary_json"]
                    continue
                candidate_memory.pop("confirmation", None)
                candidate_memory["observed_history"] = controller_history(prefix)
                candidate_memory["recent_exchange"] = {
                    "source_ids": list(range(previous, target)),
                    "owner": "controller",
                    "instruction": "Read the following original turns; they supersede older memory. Do not infer consent or execution from this pointer.",
                }
                attempt_errors = summary_errors(candidate_memory, prefix)
                text = (
                    "Original customer request:\n"
                    + raw[1]["content"]
                    + "\n\nPrior working memory (not a new customer instruction):\n<memory>"
                    + json.dumps(
                        candidate_memory, separators=(",", ":"), ensure_ascii=False
                    )
                    + "</memory>"
                )
                candidate_compact = [
                    copy.deepcopy(messages[0]),
                    {"role": "user", "content": text},
                ] + tail
                if count(candidate_compact, True) > 7680:
                    attempt_errors.append("summary_context_exceeds_7680")
                attempt_errors += [
                    "lost_argument:" + value
                    for value in lost_arguments(
                        messages[:target], candidate_compact, messages[target]
                    )
                ]
                if count(candidate_compact + [messages[target]]) > 8192:
                    attempt_errors.append("target_exceeds_8192")
                atomic(
                    folder / f"summary-before-{target}-attempt-{attempt}-checks.json",
                    {"errors": attempt_errors},
                )
                if not attempt_errors:
                    memory, compact = candidate_memory, candidate_compact
                    break
            if memory is None:
                raise ValueError(
                    f"Could not prepare context before message {target}: {attempt_errors}"
                )
            memory_path = folder / f"memory-before-{target}.json"
            atomic(memory_path, memory)
            # Add subsequent original turns causally until another compaction is needed.
            chosen = []
            end = target
            for index in remaining:
                candidate_ms = compact + messages[target : index + 1]
                if count(candidate_ms) > 8192:
                    break
                before = compact + messages[target:index]
                if lost_arguments(messages[:index], before, messages[index]):
                    break
                chosen.append(index)
                end = index
            if not chosen:
                raise ValueError("No target fits after verified memory")
            ms = compact + copy.deepcopy(messages[target : end + 1])
            mapping = (
                [0, -1] + list(range(previous, target)) + list(range(target, end + 1))
            )
            add(ms, mapping, chosen, str(memory_path.resolve()))
            audits.append(
                {
                    "target_start": target,
                    "target_end": end,
                    "prompt_tokens": count(compact, True),
                    "memory_path": str(memory_path.resolve()),
                }
            )
            remaining = [i for i in remaining if i > end]
    if covered != assistant_ids or len(set(covered)) != len(covered):
        raise ValueError("Original assistant targets were dropped or duplicated")
    for row in prepared:
        for local in row["loss_message_indices"]:
            if row["messages"][local] != messages[row["source_message_indices"][local]]:
                raise ValueError("Assistant target changed")
    return {
        "status": "ok",
        "source_episode": str(source),
        "partition": record["partition"],
        "task_id": record["task_id"],
        "source_tokens": total,
        "assistant_targets": len(covered),
        "summary_count": len(audits),
        "audits": audits,
        "rows": prepared,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/retail-data/retail-sft-final"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/retail-data/qwen-sft-parallel")
    parser.add_argument("--endpoint", action="append", help="Frozen summarizer endpoint; repeat for multiple workers")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error("Output exists; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prepared").mkdir(exist_ok=True)
    prompt = (
        (ROOT / "templates/retail_memory_summarizer.txt").read_text()
        + "\nDataset context checkpoint: recent messages can be a tool call plus its tool response, not necessarily a customer confirmation. Keep that exchange verbatim. Preserve all still-pending requests, exact database facts, names/ZIP/email needed for lookups, and already-executed actions to avoid repetition. Use other_orders/other_payments arrays with source_ids if several requests are pending. pending_action may be null if nothing remains. Do not invent the next action. The next assistant target is not provided."
    )
    config = {
        "context_limit": 8192,
        "prompt_budget": 7680,
        "prompt_sha256": digest(prompt),
        "endpoints": args.endpoint or ["http://127.0.0.1:8004/v1", "http://127.0.0.1:8005/v1"],
        "smoke": args.smoke,
    }
    if args.input_dir.resolve() != (ROOT / "outputs/retail-data/qwen-sft-parallel").resolve():
        config["input_dir"] = str(args.input_dir.resolve())
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Preparation config changed")
    atomic(config_path, config)
    (output / "summarizer-prompt.txt").write_text(prompt)
    snapshot = output / "input-snapshot.json"
    if not snapshot.exists():
        selected, excluded = [], []
        for split in ["sft", "validation"]:
            for record in read_rows(
                args.input_dir / (split + ".jsonl")
            ):
                episode = json.loads(Path(record["source_episode"]).read_text())
                original = episode["record"]
                if (
                    original.get("forced_stop")
                    or original.get("error")
                    or original.get("reward") is None
                ):
                    excluded.append(
                        {
                            "source_episode": record["source_episode"],
                            "task_id": record["task_id"],
                            "reason": "incomplete_or_infrastructure_error",
                        }
                    )
                else:
                    selected.append(record)
        if args.smoke:
            selected = [
                max(
                    (r for r in selected if r["task_id"] == task),
                    key=lambda r: r["source_tokens"],
                )
                for task in ["2", "58", "23"]
            ]
        atomic(snapshot, {"selected": selected, "excluded": excluded})
    data = json.loads(snapshot.read_text())
    pending = [
        r
        for r in data["selected"]
        if not (
            output / "prepared" / digest(r["source_episode"]) / "result.json"
        ).exists()
    ]
    queue = list(reversed(pending))
    queue_lock = threading.Lock()

    def worker(index):
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "outputs/hf-export/tiny-aya-global-function-calling-sft-525",
            local_files_only=True,
        )
        client = OpenAI(
            base_url=config["endpoints"][index],
            api_key="local",
            timeout=300,
            max_retries=2,
        )
        while True:
            with queue_lock:
                if not queue:
                    return
                record = queue.pop()
            folder = output / "prepared" / digest(record["source_episode"])
            folder.mkdir(exist_ok=True)
            try:
                result = prepare(record, folder, tokenizer, client, prompt)
            except Exception as error:  # noqa: BLE001 -- Preserve failed source and continue independent preparations.
                result = {
                    "status": "error",
                    "source_episode": record["source_episode"],
                    "task_id": record["task_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            atomic(folder / "result.json", result)
            print(
                f"worker={index} task={record['task_id']} status={result['status']} segments={len(result.get('rows', []))} summaries={result.get('summary_count')} {result.get('error', '')}",
                flush=True,
            )

    def publish(finished=False):
        results = [
            json.loads(p.read_text())
            for p in (output / "prepared").glob("*/result.json")
        ]
        pools = {"sft": [], "validation": []}
        seen = {}
        duplicates = 0
        for result in results:
            for row in result.get("rows", []):
                key = digest(row["messages"])
                if key in seen:
                    other = seen[key]
                    if (
                        other["group_id"] != row["group_id"]
                        or other["partition"] != row["partition"]
                    ):
                        raise ValueError(
                            "Identical context crosses task/split boundaries"
                        )
                    other["loss_message_indices"] = sorted(
                        set(other["loss_message_indices"])
                        | set(row["loss_message_indices"])
                    )
                    other.setdefault("additional_source_episodes", []).append(
                        row["source_episode"]
                    )
                    duplicates += 1
                else:
                    seen[key] = row
                    pools[row["partition"]].append(row)
        for split, rows in pools.items():
            random.Random(2026).shuffle(rows)
            path = output / (split + ".jsonl")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            temporary.replace(path)
        rejected = [r for r in results if r["status"] != "ok"]
        report = {
            "finished": finished,
            "input_trajectories": len(data["selected"]),
            "excluded_incomplete": data["excluded"],
            "processed": len(results),
            "successful_trajectories": sum(r["status"] == "ok" for r in results),
            "failed_trajectories": len(rejected),
            "examples": {k: len(v) for k, v in pools.items()},
            "summaries": sum(r.get("summary_count", 0) for r in results),
            "deduplicated_contexts": duplicates,
            "rejected": rejected,
            "dataset_ready": finished and not rejected,
            "updated_at": time.time(),
        }
        atomic(output / "report.json", report)
        return report

    with ThreadPoolExecutor(max_workers=len(config["endpoints"])) as executor:
        futures = [executor.submit(worker, i) for i in range(len(config["endpoints"]))]
        while not all(f.done() for f in futures):
            publish()
            wait(futures, timeout=5)
        for future in futures:
            future.result()
    report = publish(finished=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
