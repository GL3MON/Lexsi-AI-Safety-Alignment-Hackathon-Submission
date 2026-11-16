"""Causal memory/continuation experiment on saved traces; never edits training data."""

import argparse
import copy
import json
import time
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def wire(messages):
    return [
        {
            k: v
            for k, v in message.items()
            if k in {"role", "content", "tool_calls", "tool_call_id", "name"}
        }
        for message in messages
    ]


def normalized_calls(message):
    return [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"])
            if isinstance(call["function"]["arguments"], str)
            else call["function"]["arguments"],
        }
        for call in message.get("tool_calls") or []
    ]


def observations(messages):
    result = {}
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            result[index] = json.loads(message["content"])
    return result


def with_catalog(messages, tools):
    view = copy.deepcopy(messages)
    view[0]["content"] += (
        "\n\n<tools>\n"
        + json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
        + "\n</tools>"
    )
    return view


def validate_memory(memory, prefix, expected, tail_start):
    """Check critical database fields against the specific cited observations."""
    errors = []
    obs = observations(prefix)

    def cite(record, label):
        ids = record.get("source_ids", [])
        if not ids or any(type(i) is not int or i < 1 or i >= len(prefix) for i in ids):
            errors.append(label + ": invalid/missing past source IDs")
        return [obs[i] for i in ids if type(i) is int and i in obs]

    def check_source_ids(value):
        if isinstance(value, dict):
            if "source_ids" in value:
                cite(value, "record")
            for child in value.values():
                check_source_ids(child)
        elif isinstance(value, list):
            for child in value:
                check_source_ids(child)

    check_source_ids(memory)

    args = expected["arguments"]
    customer = memory.get("customer", {})
    sources = cite(customer, "customer")
    if customer.get("authenticated") is not True or not any(
        s["name"] in {"find_user_id_by_name_zip", "find_user_id_by_email"}
        and s["content"] == customer.get("user_id")
        for s in sources
    ):
        errors.append("Authentication lacks its successful lookup source")
    order = memory.get("active_order", {})
    sources = cite(order, "active_order")
    verified = [
        s["content"]
        for s in sources
        if s["name"] == "get_order_details"
        and isinstance(s["content"], dict)
        and s["content"].get("order_id") == order.get("order_id")
    ]
    if order.get("order_id") != args.get("order_id") or not any(
        o.get("status") == order.get("status")
        and o.get("user_id") == customer.get("user_id")
        for o in verified
    ):
        errors.append(
            "Active order/status/ownership does not match cited tool data and target"
        )
    for field, argument in [
        ("items", "item_ids"),
        ("replacement_items", "new_item_ids"),
    ]:
        records = memory.get(field, [])
        if not set(args.get(argument, [])) <= {r.get("item_id") for r in records}:
            errors.append(field + ": required IDs missing")
        for record in records:
            if not {"item_id", "product_id", "price", "source_ids"} <= set(record):
                errors.append(field + ": required item fields missing")
            sources = cite(record, field)
            matches = []
            for source in sources:
                body = source["content"]
                if not isinstance(body, dict):
                    continue
                if (
                    field == "items"
                    and source["name"] == "get_order_details"
                    and body.get("order_id") == order.get("order_id")
                ):
                    matches.extend(
                        item
                        for item in body.get("items", [])
                        if item.get("item_id") == record.get("item_id")
                    )
                for item in (
                    body.get("variants", {}).values()
                    if field == "replacement_items"
                    and source["name"] == "get_product_details"
                    else []
                ):
                    if item.get("item_id") == record.get("item_id"):
                        matches.append(
                            dict(
                                item,
                                product_id=body["product_id"],
                                name=body.get("name"),
                            )
                        )

            def agrees(item, record=record):
                return (
                    all(
                        record.get(k) == item.get(k)
                        for k in ["item_id", "product_id", "price", "name"]
                        if k in record
                    )
                    and all(
                        item.get("options", {}).get(k) == v
                        for k, v in record.get("options", {}).items()
                    )
                    and (
                        "available" not in record
                        or record["available"] == item.get("available")
                    )
                )

            if not any(agrees(item) for item in matches):
                errors.append(
                    field
                    + ": cited item/options/price/availability mismatch for "
                    + str(record.get("item_id"))
                )
    payment = memory.get("payment", {})
    sources = cite(payment, "payment")
    payment_id = payment.get("payment_method_id")
    valid_payment = False
    for source in sources:
        body = source["content"]
        if not isinstance(body, dict):
            continue
        method = body.get("payment_methods", {}).get(payment_id)
        if method and (
            payment.get("balance") is None
            or payment["balance"] == method.get("balance")
        ):
            valid_payment = True
        if payment.get("balance") is None and any(
            t.get("payment_method_id") == payment_id
            for t in body.get("payment_history", [])
        ):
            valid_payment = True
    if payment_id != args.get("payment_method_id") or not valid_payment:
        errors.append("Payment ID/balance lacks its cited database source")
    confirmation = memory.get("confirmation", {})
    cite(confirmation, "confirmation")
    if confirmation.get("status") not in {"confirmed", "see_recent_tail"} or not {
        tail_start,
        tail_start + 1,
    } <= set(confirmation.get("source_ids", [])):
        errors.append(
            "Confirmation must cite the preserved explanation and customer approval"
        )
    if confirmation.get("status") != "see_recent_tail" and (
        confirmation.get("order_id") != args.get("order_id")
        or not set(args.get("item_ids", [])) <= set(confirmation.get("item_ids", []))
    ):
        errors.append("Confirmation action scope is incomplete")
    if memory.get("pending_action", {}).get("executed") is not False:
        errors.append("Pending action incorrectly marked executed")
    for record in memory.get("actions_taken", []):
        sources = cite(record, "executed action")
        matches = [s for s in sources if s["name"] == record.get("name")]

        def outcome(source):
            body = source["content"]
            failed = (
                isinstance(body, str)
                and body.lstrip().lower().startswith("error")
                or isinstance(body, dict)
                and bool(body.get("error"))
            )
            return "failed" if failed else "succeeded"

        if not matches or not any(
            outcome(source) == record.get("status") for source in matches
        ):
            errors.append("Past action status lacks its matching tool response")
    for record in memory.get("preferences", []) + memory.get("resolved_requests", []):
        cite(record, "textual fact")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--task-ids", default="2,58,23")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/retail-data/summarizer-test" / (
        "run-" + uuid4().hex[:10]
    )
    output.mkdir(parents=True, exist_ok=False)
    prompt = (ROOT / "templates/retail_memory_summarizer.txt").read_text()
    (output / "prompt.txt").write_text(prompt)
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=300, max_retries=1)
    student = AutoTokenizer.from_pretrained(
        ROOT / "outputs/hf-export/tiny-aya-global-function-calling-sft-525",
        local_files_only=True,
    )
    teacher = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-4B", revision=REVISION, local_files_only=True
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "outputs/retail-data/qwen-sft-parallel/sft.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    settings = {
        "base_url": args.base_url,
        "model": "qwen3.5-4b",
        "server_profile": json.loads(
            (ROOT / "outputs/logs/serving-8005.json").read_text()
        ),
        "prompt_tokens_budget": 7680,
        "student_context_limit": 8192,
        "output_reserve": 512,
        "note": "Qwen actor proxy; no Aya GPU inference or environment execution",
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2))

    def query(messages, tools=None, summary=False):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="qwen3.5-4b",
            messages=messages,
            temperature=0,
            seed=42,
            max_tokens=2400 if summary else 512,
            **(
                {"response_format": {"type": "json_object"}}
                if summary
                else {
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.model_dump(), time.perf_counter() - start

    results = []
    for task in args.task_ids.split(","):
        candidates = sorted(
            [
                r
                for r in rows
                if r["task_id"] == task
                and r.get("source_terminal_reward") == 1
                and r["source_tokens"] > 8192
            ],
            key=lambda r: r["source_tokens"],
            reverse=True,
        )
        row = candidates[0]
        episode = json.loads(Path(row["source_episode"]).read_text())
        history = wire(episode["messages"])
        target_index = max(
            i
            for i, message in enumerate(history)
            if any(
                c["name"].startswith(("return_", "exchange_", "modify_", "cancel_"))
                for c in normalized_calls(message)
            )
        )
        expected = normalized_calls(history[target_index])[0]
        prefix = history[:target_index]
        tail_start = target_index - 2
        if prefix[tail_start]["role"] != "assistant" or prefix[-1]["role"] != "user":
            raise ValueError(
                "This experiment expects an explanation/customer-confirmation tail"
            )
        numbered = [{"message_id": i, **m} for i, m in enumerate(prefix) if i]
        payload = {
            "policy": prefix[0]["content"],
            "history_to_summarize": numbered[: tail_start - 1],
            "recent_messages_kept_verbatim": numbered[tail_start - 1 :],
            "recent_tail_message_ids": [tail_start, tail_start + 1],
        }
        summary_input = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        summary_input_tokens = len(
            teacher.apply_chat_template(
                summary_input,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if summary_input_tokens + 2400 > 16384:
            raise ValueError("Summarizer request exceeds teacher context")
        folder = output / ("task-" + task)
        folder.mkdir()
        (folder / "summary-input.json").write_text(
            json.dumps(summary_input, indent=2, ensure_ascii=False)
        )
        print(
            f"task={task} source={row['source_tokens']} summarizer_prompt={summary_input_tokens}",
            flush=True,
        )
        response, elapsed = query(summary_input, summary=True)
        (folder / "summary-response.json").write_text(json.dumps(response, indent=2))
        if response["choices"][0]["finish_reason"] == "length":
            raise RuntimeError(
                "Summary truncated; retain audit rather than accept partial JSON"
            )
        memory = json.loads(response["choices"][0]["message"]["content"])
        # Confirmation is never regenerated by the summarizer: retain the real
        # explanation/reply and attach a causal pointer, without claiming consent.
        memory["confirmation"] = {
            "status": "see_recent_tail",
            "source_ids": [tail_start, tail_start + 1],
            "owner": "controller",
        }
        (folder / "memory.json").write_text(
            json.dumps(memory, indent=2, ensure_ascii=False)
        )
        errors = validate_memory(memory, prefix, expected, tail_start)
        content = (
            "Original customer request:\n"
            + prefix[1]["content"]
            + "\n\nWorking memory extracted from prior messages (not a new customer instruction):\n<memory>"
            + json.dumps(memory, separators=(",", ":"), ensure_ascii=False)
            + "</memory>"
        )
        compact = [
            copy.deepcopy(prefix[0]),
            {"role": "user", "content": content},
        ] + copy.deepcopy(prefix[tail_start:])
        (folder / "compact-context.json").write_text(
            json.dumps(compact, indent=2, ensure_ascii=False)
        )
        tools = episode["record"]["tools"]
        student_messages = with_catalog(compact, tools)
        compact_tokens = len(
            student.apply_chat_template(
                student_messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        )
        baseline, baseline_seconds = query(prefix, tools=tools)
        continuation, continuation_seconds = query(compact, tools=tools)
        (folder / "baseline-response.json").write_text(json.dumps(baseline, indent=2))
        (folder / "compact-response.json").write_text(
            json.dumps(continuation, indent=2)
        )
        baseline_calls = normalized_calls(baseline["choices"][0]["message"])
        compact_calls = normalized_calls(continuation["choices"][0]["message"])
        result = {
            "task_id": task,
            "source_episode": row["source_episode"],
            "source_tokens": row["source_tokens"],
            "target_message_index": target_index,
            "summary_input_tokens": summary_input_tokens,
            "summary_output_tokens": response["usage"]["completion_tokens"],
            "summary_seconds": elapsed,
            "compact_student_prompt_tokens": compact_tokens,
            "original_student_prompt_tokens": len(
                student.apply_chat_template(
                    with_catalog(prefix, tools),
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                )
            ),
            "fits_with_512_output_reserve": compact_tokens <= 7680,
            "critical_fact_errors": errors,
            "recent_tail_unchanged": compact[-2:] == prefix[tail_start:],
            "expected_call": expected,
            "baseline_calls": baseline_calls,
            "compact_calls": compact_calls,
            "baseline_matches_recorded": baseline_calls == [expected],
            "compact_matches_baseline": compact_calls == baseline_calls,
            "compact_matches_recorded": compact_calls == [expected],
            "baseline_seconds": baseline_seconds,
            "compact_seconds": continuation_seconds,
            "limitations": "Critical field/source checks do not validate every natural-language claim; actor proxy is Qwen, not Aya.",
        }
        result["passed"] = (
            not errors
            and result["fits_with_512_output_reserve"]
            and result["baseline_matches_recorded"]
            and result["compact_matches_recorded"]
            and result["recent_tail_unchanged"]
        )
        results.append(result)
        (folder / "report.json").write_text(json.dumps(result, indent=2))
        (output / "report.json").write_text(
            json.dumps(
                {
                    "cases": len(results),
                    "passed": sum(r["passed"] for r in results),
                    "results": results,
                },
                indent=2,
            )
        )
        print(
            f"task={task} passed={result['passed']} compact_tokens={compact_tokens} errors={errors} same_call={result['compact_matches_recorded']}",
            flush=True,
        )
    print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
