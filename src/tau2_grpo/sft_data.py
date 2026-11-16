"""Group-safe validation and exact assistant-only labels for native chat templates."""

import json
import random
from collections import defaultdict

from tau2_grpo.prepare_function_calling import digest, model_messages, prompt_group
from tau2_grpo.tool_messages import template_supports_tool_roles

_PREPROCESS_TOKENIZER=None


def _init_preprocessing(tokenizer):
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    global _PREPROCESS_TOKENIZER
    _PREPROCESS_TOKENIZER=tokenizer


def _encode_record(payload):
    record,max_length=payload
    try:
        encoded,spans=tokenize_assistant_only(_PREPROCESS_TOKENIZER,record['messages'],max_length,record.get('loss_message_indices'))
        return record,encoded,spans,None
    except ValueError as error:return record,None,None,str(error)


def encode_records(tokenizer,records,max_length,workers=1):
    if workers==1:
        for record in records:
            try:
                encoded,spans=tokenize_assistant_only(tokenizer,record['messages'],max_length,record.get('loss_message_indices'))
                yield record,encoded,spans,None
            except ValueError as error:yield record,None,None,str(error)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_init_preprocessing,initargs=(tokenizer,)) as pool:
            yield from pool.map(_encode_record,((r,max_length) for r in records),chunksize=4)


def read_records(path):
    records = []
    seen = set()
    with path.open() as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Missing messages: {path}:{number}")
            record = dict(record, messages=[dict(m) for m in messages])
            computed_group = (
                digest({"domain": record.get("domain", "task"), "task_id": str(record["task_id"])})
                if "task_id" in record else prompt_group(record["messages"])
            )
            if record.get("group_id", computed_group) != computed_group:
                raise ValueError(f"Incorrect group_id: {path}:{number}")
            record["group_id"] = computed_group
            record["id"] = record.get("id", digest(messages))
            content_hash = digest(record["messages"])
            if content_hash in seen:
                raise ValueError(f"Duplicate conversation: {path}:{number}")
            seen.add(content_hash)
            records.append(record)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def split_validation(records, fraction, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    if len(groups) < 2:
        raise ValueError("Need at least two prompt groups for training/validation")
    strata = defaultdict(list)
    for group, members in groups.items():
        signature = tuple(sorted({sum(m["role"] == "assistant" for m in r["messages"]) for r in members}))
        strata[signature].append(group)
    rng = random.Random(seed)
    eval_groups = set()
    for signature in sorted(strata):
        ids = sorted(strata[signature])
        rng.shuffle(ids)
        count = min(len(ids) - 1, round(len(ids) * fraction))
        eval_groups.update(ids[:count])
    if not eval_groups:
        eval_groups.add(sorted(groups)[-1])
    train = [r for r in records if r["group_id"] not in eval_groups]
    evaluation = [r for r in records if r["group_id"] in eval_groups]
    rng.shuffle(train)
    rng.shuffle(evaluation)
    return train, evaluation


def validate_task_partitions(train, evaluation, manifest):
    """Enforce fixed task reservations before any tokenization or training."""
    partitions = manifest["partitions"]
    pools = list(partitions.values())
    if any(set(a) & set(b) for i, a in enumerate(pools) for b in pools[i + 1:]):
        raise ValueError("Task manifest partitions overlap")
    for name, records, allowed in (
        ("train", train, set(partitions["sft"])),
        ("eval", evaluation, set(partitions["validation"])),
    ):
        if any(str(r.get("task_id", "")) not in allowed for r in records):
            raise ValueError(f"{name} data contains a missing or unreserved task_id")
    if {r["group_id"] for r in train} & {r["group_id"] for r in evaluation}:
        raise ValueError("Training and evaluation prompt groups overlap")


def tokenize_assistant_only(tokenizer, messages, max_length, loss_message_indices=None):
    """Derive response spans from exact token prefixes, never substring matching.

    Every prefix must match the full conversation. This intentionally fails on
    templates that rewrite earlier turns rather than silently masking wrong tokens.
    Assistant end-of-turn tokens are included so the model learns when to stop.
    """
    # Modified Aya Jinja handles tool-role rendering; legacy templates keep their adapter.
    messages = messages if template_supports_tool_roles(tokenizer) else model_messages(messages)
    selected = None if loss_message_indices is None else set(loss_message_indices)
    if selected is not None and (
        not selected or any(type(i) is not int or i < 0 or i >= len(messages) or messages[i]["role"] != "assistant" for i in selected)
    ):
        raise ValueError("loss_message_indices must identify assistant messages")

    def encode(prefix, generation=False):
        return tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=generation, return_dict=False,
        )

    ids = encode(messages)
    if len(ids) > max_length:
        raise ValueError(f"overlength:{len(ids)}>{max_length}")
    labels = [-100] * len(ids)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = encode(messages[:index], generation=True)
        complete = encode(messages[:index + 1])
        if not prompt or complete[:len(prompt)] != prompt or ids[:len(complete)] != complete:
            raise ValueError("Chat template is not prefix-preserving; assistant loss boundaries cannot be verified")
        start, end = len(prompt), len(complete)
        if end <= start:
            raise ValueError("Assistant turn has no target tokens")
        if selected is None or index in selected:
            labels[start:end] = ids[start:end]
        spans.append([start, end])
    if not spans or not any(label != -100 for label in labels[1:]):
        raise ValueError("Conversation has no trainable assistant tokens")
    return {"input_ids": ids, "labels": labels}, spans


def tool_call_count(record):
    """Count assistant calls only; ignore tool results and schema examples."""
    from tau2_grpo.prepare_function_calling import parse_calls
    return sum(len(parse_calls(m["content"])) for m in record["messages"] if m["role"] == "assistant")


def order_by_tool_calls(records):
    # Seeded tie ordering avoids sorting a whole count bucket by source IDs.
    rows = list(records)
    random.Random(42).shuffle(rows)
    return sorted(rows, key=tool_call_count)
