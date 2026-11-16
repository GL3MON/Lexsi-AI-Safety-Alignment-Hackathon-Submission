"""Audit deterministic context selection for one retail mutation, without editing datasets."""

import argparse
import copy
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.sft_data import tokenize_assistant_only

ROOT = Path(__file__).resolve().parents[1]


def select_context(messages, target_index):
    """Select evidence using only history, with no generated summary or future results."""
    history = messages[:target_index]
    customers = [
        i
        for i, m in enumerate(history)
        if m["role"] == "user" and not m["content"].startswith("<tool_response>")
    ]
    last_customer = customers[-1]
    order_ids = re.findall(r"#W\d+", history[last_customer]["content"])
    if not order_ids:
        raise ValueError("Latest customer turn does not identify the active order")
    active_order = order_ids[-1]
    observations = []
    for index, message in enumerate(history):
        match = re.fullmatch(
            r"<tool_response>\s*(.*?)\s*</tool_response>", message["content"], re.DOTALL
        )
        if match:
            observations.append((index, json.loads(match.group(1))))
    order_index, order = next(
        (i, obj["content"])
        for i, obj in reversed(observations)
        if obj["name"] == "get_order_details"
        and isinstance(obj["content"], dict)
        and obj["content"].get("order_id") == active_order
    )
    user_id = order["user_id"]
    user_index, user = next(
        (i, obj["content"])
        for i, obj in reversed(observations)
        if obj["name"] == "get_user_details"
        and isinstance(obj["content"], dict)
        and obj["content"].get("user_id") == user_id
    )
    auth_index, _ = next(
        (i, obj)
        for i, obj in reversed(observations)
        if obj["name"] in {"find_user_id_by_email", "find_user_id_by_name_zip"}
        and obj["content"] == user_id
    )
    keep = {0, customers[0], last_customer, target_index}
    reasons = {}

    def pair(index, reason):
        call = parse_calls(history[index - 1]["content"])
        if len(call) != 1:
            raise ValueError("Expected one preceding tool call")
        keep.update([index - 1, index])
        reasons[str(index)] = reason

    pair(auth_index, "Authentication result")
    pair(user_index, "User ownership, orders, payment methods and gift-card balance")
    pair(order_index, "Active order status, original items, options and prices")
    auth_call = parse_calls(history[auth_index - 1]["content"])[0]
    for value in auth_call["arguments"].values():
        matches = [
            i
            for i in customers
            if i < auth_index
            and str(value).casefold() in history[i]["content"].casefold()
        ]
        if not matches:
            raise ValueError("Authentication input has no customer source")
        keep.add(matches[-1])
    products = {item["product_id"] for item in order["items"]}
    latest_products = {}
    for index, obj in observations:
        body = obj["content"]
        if (
            obj["name"] == "get_product_details"
            and isinstance(body, dict)
            and body.get("product_id") in products
        ):
            latest_products[body["product_id"]] = (index, body)
    for index, _ in latest_products.values():
        pair(index, "Full product variants, prices and availability")
    # Preserve all customer constraints expressed after inspecting the active order.
    keep.update(i for i in customers if i > order_index)
    # Preserve the original explanation immediately preceding the latest confirmation.
    if history[last_customer - 1]["role"] != "assistant":
        raise ValueError("No assistant explanation before customer confirmation")
    keep.add(last_customer - 1)
    selected = [copy.deepcopy(messages[i]) for i in sorted(keep)]
    evidence = {
        "active_order": order,
        "user": user,
        "products": [body for _, body in latest_products.values()],
    }
    return selected, sorted(keep), reasons, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.candidate.read_text().splitlines()[0])
    messages = copy.deepcopy(row["messages"])
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
    target_index = max(
        i
        for i, message in enumerate(messages)
        if message["role"] == "assistant"
        and any(
            call["name"].startswith(("exchange_", "return_", "cancel_", "modify_"))
            for call in parse_calls(message["content"])
        )
    )
    selected, indices, reasons, evidence = select_context(messages, target_index)
    target = parse_calls(selected[-1]["content"])[0]
    arguments = target["arguments"]
    order = evidence["active_order"]
    checks = {
        "original_target_unchanged": selected[-1] == messages[target_index],
        "system_policy_and_tools_unchanged": selected[0] == messages[0],
        "only_history_before_target_used": all(i <= target_index for i in indices),
        "order_matches_latest_customer": arguments["order_id"] == order["order_id"],
        "delivered_status_preserved": order["status"] == "delivered",
        "original_items_preserved": set(arguments["item_ids"])
        <= {item["item_id"] for item in order["items"]},
        "payment_method_preserved": arguments["payment_method_id"]
        in evidence["user"]["payment_methods"],
    }
    variants = {
        key: value
        for product in evidence["products"]
        for key, value in product["variants"].items()
    }
    checks["new_items_available"] = all(
        item in variants and variants[item]["available"]
        for item in arguments.get("new_item_ids", [])
    )
    if not all(checks.values()):
        raise ValueError(checks)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "outputs/hf-export/tiny-aya-global-function-calling-sft-525",
        local_files_only=True,
    )
    count = lambda value: len(
        tokenizer.apply_chat_template(value, tokenize=True, return_dict=False)
    )
    report = {
        "source_episode": row["source_episode"],
        "task_id": row["task_id"],
        "original_trajectory_tokens": count(messages),
        "original_context_through_target_tokens": count(messages[: target_index + 1]),
        "pruned_example_tokens": count(selected),
        "fits_8192": count(selected) <= 8192,
        "target_original_message_index": target_index,
        "kept_original_message_indices": indices,
        "protected_evidence": reasons,
        "checks": checks,
        "training_target": target,
        "scope": "Single target assistant action; earlier assistant turns are context only. Not a production pruner or a policy-quality verdict.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if report["fits_8192"]:
        encoded, spans = tokenize_assistant_only(tokenizer, selected, 8192)
        target_span = spans[-1]
        report["assistant_span_format"] = target_span
        # Repeated history must not become duplicated training supervision.
        start, end = target_span
        encoded["labels"] = [
            label if start <= i < end else -100
            for i, label in enumerate(encoded["labels"])
        ]
        report["supervised_tokens"] = sum(label != -100 for label in encoded["labels"])
        (args.output / "tokenized-example.json").write_text(json.dumps(encoded))
    (args.output / "pruned-messages.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
