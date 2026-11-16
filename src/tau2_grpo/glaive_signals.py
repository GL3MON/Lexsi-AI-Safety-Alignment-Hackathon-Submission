"""Deterministic dense signals for reference calls and controlled clarification."""
import json
import re
from collections import Counter
from tau2_grpo.glaive_rewards import call_key, score_calls
from tau2_grpo.prepare_function_calling import parse_calls, tool_validators, CALL_RE


def score_decision(text, row):
    reference = row['reward_reference']
    def unique_object(pairs):
        obj={}
        for key,value in pairs:
            if key in obj: raise ValueError('duplicate_json_key')
            obj[key]=value
        return obj
    def invalid_constant(value): raise ValueError('nonfinite_json_number')
    try:
        calls = parse_calls(text)
        for block in CALL_RE.findall(text):
            json.loads(block, object_pairs_hook=unique_object, parse_constant=invalid_constant)
        valid_format = '<tool_response' not in text
    except ValueError:
        calls = []
        valid_format = False
    if reference.get('decision') == 'clarify':
        # This checks a targeted information request, not general NL quality.
        aliases = reference['missing_field_aliases']
        requests = re.findall(r'[^.!?\n]*\?|[^.!?\n]*\b(?:please|kindly)\s+(?:provide|share|tell|enter|confirm)[^.!?\n]*', text.lower())
        question = ' '.join(requests)
        targeted = any(re.search(r'(?<!\w)'+re.escape(a.lower())+r'(?!\w)', question) for a in aliases)
        asks = valid_format and not calls and '<tool_call' not in text and bool(question) and targeted
        return {'reward': float(asks), 'clarification_correct': float(asks),
                'call_precision': 0., 'tool_selection': 0., 'argument_accuracy': 0.,
                'schema_validity': 0., 'copy_accuracy': 0., 'redundant_call_fraction': 0.,
                'format_valid': float(valid_format), 'reward_version': 'glaive-signals-v1'}
    gold = reference['tool_calls']
    result = score_calls(text, gold)
    if not valid_format or result['parse_error']:
        return {**result, 'reward': 0., 'tool_selection': 0., 'argument_accuracy': 0.,
                'schema_validity': 0., 'copy_accuracy': 0., 'redundant_call_fraction': 0.,
                'format_valid': 0., 'reward_version': 'glaive-signals-v1'}
    validators = tool_validators(row['tools'])
    remaining = list(gold)
    selection = args_correct = args_total = copies_correct = copies_total = schema = 0
    evidence = reference.get('literal_arguments', {})
    for call in calls:
        validator = validators.get(call['name'])
        unknown = set(call['arguments']) - set(validator.schema.get('properties', {})) if validator else set()
        schema += int(validator is not None and not unknown and validator.is_valid(call['arguments']))
        candidates = [(i,g) for i,g in enumerate(remaining) if g['name']==call['name']]
        if not candidates: continue
        # Allocate each reference at most once; prefer the closest argument match.
        i,g = max(candidates, key=lambda pair: sum(call_key(v)==call_key(call['arguments'].get(k)) for k,v in pair[1]['arguments'].items()))
        remaining.pop(i); selection += 1
        keys = set(g['arguments']) | set(call['arguments'])
        args_total += len(keys)
        args_correct += sum(k in g['arguments'] and k in call['arguments'] and call_key(g['arguments'][k])==call_key(call['arguments'][k]) for k in keys)
        for key,value in evidence.get(call['name'], {}).items():
            copies_total += 1
            copies_correct += int(key in call['arguments'] and call_key(value)==call_key(call['arguments'][key]))
    n=max(len(calls),len(gold),1)
    tool_score=selection/n
    arg_score=(args_correct/args_total if args_total else float(selection>0))*tool_score
    schema_score=schema/max(len(calls),1)
    copy_score=(copies_correct/copies_total if copies_total else 0.)*tool_score
    counts=Counter(call_key(c) for c in calls)
    redundant=sum((counts-Counter(call_key(g) for g in gold)).values())/max(len(calls),1)
    # No standalone format/schema bonus: valid but irrelevant calls earn nothing.
    reward = max(-.1, .7*result['call_precision'] + .1*tool_score + .15*arg_score + .05*(copy_score if copies_total else arg_score) - .1*redundant)
    if not calls: reward=0.
    return {**result,'reward': reward,'tool_selection':tool_score,'argument_accuracy':arg_score,
            'schema_validity':schema_score,'copy_accuracy':copy_score,'redundant_call_fraction':redundant,
            'format_valid':1.,'reward_version':'glaive-signals-v1'}
