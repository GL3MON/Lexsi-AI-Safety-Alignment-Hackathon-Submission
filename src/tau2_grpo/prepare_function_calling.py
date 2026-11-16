#!/usr/bin/env python3
"""Prepare Hermes function-calling SFT, RL and test data from a pinned source.

Default: Glaive initial assistant tool-call decisions, with group-disjoint
SFT training/validation, RL training, and final test partitions.
Legacy source remains available with --source original.

Outputs are structured JSONL; model-specific chat-template rendering and
assistant-only loss masks belong in the SFT trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "NousResearch/hermes-function-calling-v1"
REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
SOURCE_FILES = {
    "glaive": ("glaive-function-calling-5k.json",),
    "original": ("func-calling-singleturn.json", "func-calling.json"),
}
ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
CATALOG_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompt_group(messages):
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    canonical = " ".join(unicodedata.normalize("NFKC", first_user).casefold().split())
    return digest(canonical)


def parse_calls(text):
    blocks = CALL_RE.findall(text)
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        raise ValueError("unbalanced_tool_call_tags")
    calls = []
    for block in blocks:
        try:
            call = json.loads(block)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid_tool_call_json") from error
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
            raise ValueError("invalid_tool_call_shape")
        calls.append(call)
    return calls


def parse_row(row):
    try:
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        messages = [{"role": ROLES[m["from"]], "content": m["value"]} for m in row["conversations"]]
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("invalid_record") from error
    if (not isinstance(tools, list) or len(messages) < 3
            or messages[0]["role"] != "system" or messages[1]["role"] != "user"
            or any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages)):
        raise ValueError("invalid_conversation_structure")
    return messages, tools


def tool_validators(tools):
    from jsonschema import Draft7Validator, SchemaError
    validators = {}
    try:
        for tool in tools:
            function = tool["function"]
            name = function["name"]
            if tool["type"] != "function" or not isinstance(name, str) or name in validators:
                raise ValueError("invalid_tool_schema")
            schema = function["parameters"]
            Draft7Validator.check_schema(schema)
            validators[name] = Draft7Validator(schema)
    except (KeyError, TypeError, SchemaError) as error:
        raise ValueError("invalid_tool_schema") from error
    return validators


def check_embedded_catalog(system_text, tools):
    # Only JSON catalog blocks count; documentation may mention literal delimiters.
    embedded = []
    for block in CATALOG_RE.findall(system_text):
        try:
            embedded.append(json.loads(block))
        except json.JSONDecodeError:
            pass
    if (tools and tools not in embedded) or (not tools and any(embedded)):
        raise ValueError("system_tool_catalog_mismatch")


def validate_calls(calls, validators):
    from jsonschema import ValidationError
    for call in calls:
        if call["name"] not in validators:
            raise ValueError("unknown_tool_target")
        try:
            validators[call["name"]].validate(call["arguments"])
        except ValidationError as error:
            raise ValueError("schema_invalid_arguments") from error


def normalize_glaive(row, source_file):
    """Use only first system/user/assistant decision, before any tool feedback."""
    messages, tools = parse_row(row)
    if messages[2]["role"] != "assistant":
        raise ValueError("first_assistant_not_initial")
    if not tools:
        raise ValueError("missing_tools")
    validators = tool_validators(tools)
    check_embedded_catalog(messages[0]["content"], tools)
    target = messages[2]["content"]
    calls = parse_calls(target)
    if not calls:
        raise ValueError("first_assistant_has_no_call")
    validate_calls(calls, validators)
    if CALL_RE.sub("", target).strip():
        raise ValueError("mixed_prose_and_tool_calls")
    first = messages[:3]
    return {
        "id": digest(first), "group_id": prompt_group(first),
        "messages": first, "tools": tools,
        "turn_type": "single_turn", "assistant_turns": 1,
        "tool_calls": len(calls), "reference_tool_calls": calls,
        "category": row.get("category"), "subcategory": row.get("subcategory"),
        "task": row.get("task"),
        "sources": [{"file": source_file, "id": row.get("id")}],
    }


def normalize_original(row, source_file):
    """Preserve the original script's full-conversation validation."""
    messages, tools = parse_row(row)
    if messages[-1]["role"] != "assistant":
        raise ValueError("invalid_conversation_structure")
    validators = tool_validators(tools)
    check_embedded_catalog(messages[0]["content"], tools)
    pending, call_count = [], 0
    for previous, message in zip(messages, messages[1:]):
        role = message["role"]
        if role == "assistant":
            if previous["role"] not in {"user", "tool"}:
                raise ValueError("invalid_role_order")
            calls = parse_calls(message["content"])
            validate_calls(calls, validators)
            pending = [c["name"] for c in calls]
            call_count += len(calls)
        elif role == "tool":
            if previous["role"] != "assistant" or not pending:
                raise ValueError("orphan_tool_response")
            blocks = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", message["content"], re.S)
            if (len(blocks) != len(pending)
                or message["content"].count("<tool_response>") != len(blocks)
                or message["content"].count("</tool_response>") != len(blocks)):
                raise ValueError("tool_response_count_mismatch")
            try:
                responses = [json.loads(b) for b in blocks]
                if [r["name"] for r in responses] != pending:
                    raise ValueError("tool_response_name_mismatch")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("invalid_tool_response_json") from error
            pending = []
        elif role == "user":
            if previous["role"] not in {"system", "assistant"}:
                raise ValueError("invalid_role_order")
        else:
            raise ValueError("unexpected_system_message")
    n_assistant = sum(m["role"] == "assistant" for m in messages)
    return {
        "id": digest(messages), "group_id": prompt_group(messages),
        "messages": messages, "tools": tools,
        "turn_type": "multi_turn" if n_assistant > 1 else "single_turn",
        "assistant_turns": n_assistant, "tool_calls": call_count,
        "category": row.get("category"), "subcategory": row.get("subcategory"),
        "task": row.get("task"),
        "sources": [{"file": source_file, "id": row.get("id")}],
    }


def model_messages(messages, model_format = None):
    if model_format == "hermes":
        # Glaive's tagged <tool_call> content remains verbatim, in assistant role.
        # Tool observations (original source only) remain tool-role messages.
        return messages
    # Backwards compatibility with the original Tiny Aya input convention.
    out = []
    for message in messages:
        if message["role"] != "tool":
            out.append(message)
            continue
        content = message["content"]
        if "<tool_response>" not in content:
            try:
                payload = json.loads(content)
            except (ValueError, TypeError):
                payload = content
            if not (isinstance(payload, dict) and "name" in payload and "content" in payload):
                if not message.get("name"):
                    raise ValueError("Untagged tool observation requires its tool name")
                payload = {"name": message["name"], "content": payload}
            content = "<tool_response>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_response>"
        out.append(dict(message, role="user", content=content))
    return out


def rl_decisions(record, model_format):
    messages = record["messages"]
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            yield {
                "id": f"{record['id']}:{index}", "conversation_id": record["id"],
                "group_id": record["group_id"], "turn_type": record["turn_type"],
                "prompt": model_messages(messages[:index], model_format),
                "tools": record["tools"],
                "reward_reference": {"assistant": message["content"],
                                     "tool_calls": parse_calls(message["content"])},
                "context_mode": "initial_decision" if index == 2 else "recorded_prefix",
            }


def partition(records, seed, source, rl_fraction):
    groups = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)
    if source == "glaive":
        ids = sorted(groups)
        random.Random(seed).shuffle(ids)
        bounds = [round(len(ids)*p) for p in (0.70, 0.80, 0.95)]
        names = {
            "sft_train": ids[:bounds[0]],
            "sft_val": ids[bounds[0]:bounds[1]],
            "rl_train": ids[bounds[1]:bounds[2]],
            "test": ids[bounds[2]:],
        }
    else:
        # Preserve the original group-stratified SFT/RL split for legacy source.
        strata = defaultdict(list)
        for group_id, members in groups.items():
            sig = tuple(sorted({(r["turn_type"], bool(r["tool_calls"])) for r in members}))
            strata[sig].append(group_id)
        assignments = {}
        rng = random.Random(seed)
        for sig in sorted(strata):
            ids = sorted(strata[sig]); rng.shuffle(ids)
            n_rl = round(len(ids)*rl_fraction)
            for i, gid in enumerate(ids):
                assignments[gid] = "rl" if i < n_rl else "sft"
        names = {name: [gid for gid in sorted(groups) if assignments[gid] == name]
                 for name in ("sft", "rl")}
    if any(not gids for gids in names.values()):
        raise RuntimeError("At least one output split is empty")
    return {name: [r for gid in gids for r in groups[gid]] for name, gids in names.items()}


def audit_splits(pools):
    ids_seen, groups_seen, prefixes_seen = set(), set(), set()
    for name, pool in pools.items():
        ids = {r["id"] for r in pool}
        groups = {r["group_id"] for r in pool}
        prefixes = {digest(r["messages"][1:i]) for r in pool
                    for i, m in enumerate(r["messages"]) if m["role"] == "assistant"}
        if ids_seen & ids or groups_seen & groups or prefixes_seen & prefixes:
            raise RuntimeError(f"Cross-split leakage detected at {name}")
        ids_seen |= ids; groups_seen |= groups; prefixes_seen |= prefixes
    return {"conversation_ids": 0, "prompt_groups": 0, "assistant_prefixes": 0}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_FILES), default="glaive")
    parser.add_argument("--output", type=Path, default=Path("outputs/hermes-glaive"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl-fraction", type=float, default=0.2,
                        help="Only used by --source original; Glaive uses 70/10/15/5")
    parser.add_argument("--model", default="NousResearch/Hermes-3-Llama-3.1-8B")
    parser.add_argument("--format", choices=("hermes", "tiny-aya"), default="hermes")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--skip-token-check", action="store_true",
                        help="Avoid tokenizer download; not recommended for training data")
    args = parser.parse_args()
    if not 0 < args.rl_fraction < 1 or args.max_tokens < 1:
        parser.error("Need 0 < rl-fraction < 1 and positive max-tokens")
    if args.output.exists():
        parser.error(f"Output exists; choose a new path: {args.output}")
    from huggingface_hub import hf_hub_download
    tokenizer, tokenizer_revision = None, None
    if not args.skip_token_check:
        from transformers import AutoConfig, AutoTokenizer
        tokenizer_revision = AutoConfig.from_pretrained(args.model)._commit_hash
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=tokenizer_revision)

    records, excluded, source_counts = {}, [], {}
    for filename in SOURCE_FILES[args.source]:
        path = Path(hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION))
        raw_bytes = path.read_bytes()
        rows = json.loads(raw_bytes)
        source_counts[filename] = {"rows": len(rows), "sha256": hashlib.sha256(raw_bytes).hexdigest()}
        for i, row in enumerate(rows):
            try:
                if not isinstance(row, dict):
                    raise ValueError("invalid_record")
                rec = (normalize_glaive if args.source == "glaive" else normalize_original)(row, filename)
                formatted_messages = model_messages(rec["messages"], args.format)
                if tokenizer is not None:
                    tokens = tokenizer.apply_chat_template(formatted_messages, tokenize=True,
                                                           add_generation_prompt=False)
                    rec["token_count"] = len(tokens)
                    if len(tokens) > args.max_tokens:
                        raise ValueError("exceeds_context_limit")
                if rec["id"] in records:
                    records[rec["id"]]["sources"].extend(rec["sources"])
                    excluded.append({"source": filename, "index": i,
                                     "source_id": row.get("id"), "reason": "duplicate_conversation"})
                else:
                    records[rec["id"]] = rec
            except (KeyError, ValueError, TypeError) as error:
                excluded.append({"source": filename, "index": i,
                                 "source_id": row.get("id"), "reason": str(error)[:200]})
        print(f"Inspected {filename}: {len(rows)} rows", flush=True)
    records = sorted(records.values(), key=lambda r: r["id"])
    pools = partition(records, args.seed, args.source, args.rl_fraction)
    overlap = audit_splits(pools)
    args.output.mkdir(parents=True)

    # Keep the original output naming scheme available for legacy callers.
    if args.source == "original":
        sft = [dict(r, messages=model_messages(r["messages"], args.format)) for r in pools["sft"]]
        rl = [d for r in pools["rl"] for d in rl_decisions(r, args.format)]
        write_jsonl(args.output / "sft.jsonl", sft)
        write_jsonl(args.output / "rl_conversations.jsonl", pools["rl"])
        write_jsonl(args.output / "rl.jsonl", rl)
    else:
        for name, pool in pools.items():
            if name.startswith("sft_"):
                output = [{k: r[k] for k in ("id", "group_id", "messages", "tools",
                                                 "category", "subcategory", "task")}
                          for r in pool]
            else:
                output = [dict(next(rl_decisions(r, args.format)),
                               category=r["category"], subcategory=r["subcategory"],
                               task=r["task"]) for r in pool]
            write_jsonl(args.output / f"{name}.jsonl", output)
    write_jsonl(args.output / "excluded.jsonl", excluded)
    write_jsonl(args.output / "split_manifest.jsonl", [
        {"id": r["id"], "group_id": r["group_id"], "split": name,
         "sources": r["sources"]}
        for name, pool in pools.items() for r in pool
    ])
    summary = {
        "dataset": DATASET, "revision": REVISION, "source": args.source,
        "source_files": source_counts, "seed": args.seed,
        "model": args.model, "tokenizer_revision": tokenizer_revision,
        "format": args.format, "max_tokens": args.max_tokens,
        "split_unit": "normalized first user request",
        "splits": {name: {
            "conversations": len(pool), "groups": len({r["group_id"] for r in pool}),
            "turn_types": dict(Counter(r["turn_type"] for r in pool)),
            "tool_calls": sum(r["tool_calls"] for r in pool),
            "max_tokens": max((r.get("token_count", 0) for r in pool), default=0),
        } for name, pool in pools.items()},
        "cross_split_overlap": overlap,
        "excluded_source_rows": dict(Counter(r["reason"] for r in excluded)),
        "notes": [
            "Glaive trains on first assistant function-call decision only.",
            "SFT targets retain source <tool_call> tags; use assistant-only loss in trainer.",
            "RL/test gold calls are separate reward_reference, never part of prompt.",
            "Prompt-disjoint by normalized initial user request, not tool-disjoint.",
            "Original source preserves the previous two-pool file interface.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()