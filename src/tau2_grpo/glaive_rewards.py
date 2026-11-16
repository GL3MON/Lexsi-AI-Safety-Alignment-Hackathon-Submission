"""Verifiable reference-call precision for offline Glaive RL decisions.

References are used by the verifier only, never appended to model prompts.
Exact JSON comparison ignores object key order, but preserves string case,
leading zeros, list order and JSON value types (including bool vs number).
"""
import json
from collections import Counter
from tau2_grpo.prepare_function_calling import parse_calls


def call_key(call):
    return json.dumps(call, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def score_calls(completion, reference_calls):
    """One-to-one multiset matching. Malformed output receives zero credit."""
    if not reference_calls:
        raise ValueError('This reward requires at least one reference call')
    expected = Counter(call_key(c) for c in reference_calls)
    error = None
    try:
        generated = parse_calls(completion)
        actual = Counter(call_key(c) for c in generated)
        # Never accept student-fabricated tool observations.
        if '<tool_response' in completion:
            raise ValueError('fabricated_tool_response')
    except (ValueError, TypeError) as exc:
        generated = []
        actual = Counter()
        error = str(exc)
    matched = sum((actual & expected).values()) if error is None else 0
    count = len(generated) if error is None else max(1, completion.count('<tool_call>'))
    precision = matched / count if count else 0.0
    recall = matched / len(reference_calls)
    return {
        'reward': precision,
        'call_precision': precision,
        'call_recall': recall,
        'call_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'exact_call_set': error is None and actual == expected,
        'matched_calls': matched,
        'generated_calls': count,
        'required_calls': len(reference_calls),
        'parse_error': error,
    }
