"""Export overlapping 8K conversation chunks from full saved retail candidates."""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from tau2_grpo.prepare_function_calling import digest, parse_calls
from tau2_grpo.sft_data import tokenize_assistant_only

ROOT = Path(__file__).resolve().parents[1]


def split_messages(messages, count, max_length=8192, overlap_tokens=1024):
    """Keep tool exchanges whole, repeat system prefix, and supervise each turn once."""
    if messages[0]["role"] != "system":
        raise ValueError("Expected policy/tool system prefix")
    blocks = []
    index = 1
    while index < len(messages):
        end = index + 1
        message = messages[index]
        if message["role"] == "assistant" and parse_calls(message["content"]):
            while end < len(messages) and messages[end]["content"].lstrip().startswith(
                "<tool_response>"
            ):
                end += 1
        elif message["content"].lstrip().startswith("<tool_response>"):
            raise ValueError("Orphan tool observation")
        blocks.append(list(range(index, end)))
        index = end
    cache = {}

    def indices(start, end):
        body = [i for block in blocks[start:end] for i in block]
        prefix = [0]
        # Aya requires the first non-system turn to be user. Retain the real
        # initial request when a later window begins with an assistant/tool pair.
        if body and messages[body[0]]["role"] == "assistant":
            if messages[1]["role"] != "user":
                raise ValueError("No initial user request available for chunk prefix")
            prefix.append(1)
        return prefix + body

    def size(start, end):
        key = (start, end)
        if key not in cache:
            cache[key] = count([messages[i] for i in indices(start, end)])
        return cache[key]

    chunks = []
    for index in range(len(blocks)):
        if size(index, index + 1) > max_length:
            raise ValueError("A single exchange plus policy/tools exceeds max_length")
    supervised = set()
    start = 0
    new_start = 0
    while new_start < len(blocks):
        # Trim overlap if it prevents the next new exchange from fitting.
        while start < new_start and size(start, new_start + 1) > max_length:
            start += 1
        if size(start, new_start + 1) > max_length:
            raise ValueError("A single exchange plus policy/tools exceeds max_length")
        end = new_start + 1
        while end < len(blocks) and size(start, end + 1) <= max_length:
            end += 1
        source_indices = indices(start, end)
        targets = [
            i
            for i in source_indices
            if messages[i]["role"] == "assistant" and i not in supervised
        ]
        if not targets:
            raise ValueError("Chunk has no new assistant supervision")
        chunks.append(
            {
                "messages": [copy.deepcopy(messages[i]) for i in source_indices],
                "source_message_indices": source_indices,
                "loss_message_indices": [
                    j for j, i in enumerate(source_indices) if i in targets
                ],
            }
        )
        supervised.update(targets)
        new_start = end
        start = end
        while (
            start > 0 and size(start - 1, end) - count([messages[0]]) <= overlap_tokens
        ):
            start -= 1
    expected = {
        i for i, message in enumerate(messages) if message["role"] == "assistant"
    }
    if supervised != expected:
        raise ValueError("Assistant supervision coverage changed")
    return chunks


def missing_evidence(chunk):
    """Flag ID arguments whose earlier customer/tool source fell outside this chunk."""
    observed = ""
    missing = []
    targets = set(chunk["loss_message_indices"])
    for index, message in enumerate(chunk["messages"]):
        if index in targets:
            for call in parse_calls(message["content"]):
                for key, value in call["arguments"].items():
                    if not key.endswith(("_id", "_ids")):
                        continue
                    for item in value if isinstance(value, list) else [value]:
                        if (
                            item != "original_payment"
                            and str(item).casefold() not in observed.casefold()
                        ):
                            missing.append(
                                {
                                    "source_message_index": chunk[
                                        "source_message_indices"
                                    ][index],
                                    "tool": call["name"],
                                    "argument": key,
                                    "value": item,
                                }
                            )
        if message["role"] in {"user", "tool"}:
            observed += "\n" + message["content"]
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/retail-data/qwen-sft-parallel/sft.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-episode", help="Optional single raw episode to demonstrate"
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--overlap-tokens", type=int, default=1024)
    parser.add_argument(
        "--tokenizer",
        default=str(
            ROOT / "outputs/hf-export/tiny-aya-global-function-calling-sft-525"
        ),
    )
    args = parser.parse_args()
    if args.max_length < 1 or args.overlap_tokens < 0:
        parser.error("Invalid length or overlap")
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    count = lambda messages: len(
        tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    )
    selected = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    if args.source_episode:
        selected = [
            r
            for r in selected
            if Path(r["source_episode"]).resolve()
            == Path(args.source_episode).resolve()
        ]
    if not selected:
        parser.error("No selected trajectories matched")
    output = []
    audits = []
    for record in selected:
        source = Path(record["source_episode"])
        candidate_path = (
            source.parents[2]
            / "exports-all-generations"
            / source.parent.name
            / (record["partition"] + "-candidates.jsonl")
        )
        full = next(
            json.loads(line)
            for line in candidate_path.read_text().splitlines()
            if json.loads(line)["source_episode"] == str(source)
        )
        messages = full["messages"]
        for message in messages:
            for tag in ("tools", "tool_call", "tool_response"):
                message["content"] = re.sub(
                    "<" + tag + r">\s*(.*?)\s*</" + tag + ">",
                    lambda match, tag=tag: (
                        "<"
                        + tag
                        + ">"
                        + json.dumps(
                            json.loads(match.group(1)),
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
        chunks = split_messages(messages, count, args.max_length, args.overlap_tokens)
        audit = {
            "source_episode": str(source),
            "task_id": record["task_id"],
            "original_tokens": count(messages),
            "chunks": [],
        }
        for index, chunk in enumerate(chunks):
            encoded, _ = tokenize_assistant_only(
                tokenizer,
                chunk["messages"],
                args.max_length,
                chunk["loss_message_indices"],
            )
            row = {
                k: v
                for k, v in record.items()
                if k not in {"messages", "id", "quality_checks"}
            }
            row.update(chunk)
            row.update(
                id=digest(
                    {
                        "source": str(source),
                        "chunk": index,
                        "indices": chunk["source_message_indices"],
                    }
                ),
                example_type="overlapping_conversation_chunk",
                chunk_index=index,
                chunk_count=len(chunks),
                source_tokens=count(messages),
                student_tokens=len(encoded["input_ids"]),
                supervised_tokens=sum(x != -100 for x in encoded["labels"]),
                supervised_assistant_turns=len(chunk["loss_message_indices"]),
                source_assistant_turns=sum(m["role"] == "assistant" for m in messages),
                supervised_tool_calls=sum(
                    len(parse_calls(chunk["messages"][i]["content"]))
                    for i in chunk["loss_message_indices"]
                ),
                context_audit={
                    "missing_argument_evidence": missing_evidence(chunk),
                    "policy_prerequisites": "not_automatically_verified",
                },
            )
            output.append(row)
            audit["chunks"].append(
                {
                    "chunk_index": index,
                    "tokens": row["student_tokens"],
                    "supervised_assistant_turns": row["supervised_assistant_turns"],
                    "context_audit": row["context_audit"],
                }
            )
        audits.append(audit)
        print(
            f"task={record['task_id']} source_tokens={count(messages)} chunks={[x['tokens'] for x in audit['chunks']]}",
            flush=True,
        )
    args.output.mkdir(parents=True)
    (args.output / "chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    )
    report = {
        "trajectories": len(selected),
        "chunks": len(output),
        "chunks_per_trajectory": dict(Counter(len(a["chunks"]) for a in audits)),
        "chunks_missing_argument_evidence": sum(
            bool(row["context_audit"]["missing_argument_evidence"]) for row in output
        ),
        "traces": audits,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "traces"}, indent=2))


if __name__ == "__main__":
    main()
