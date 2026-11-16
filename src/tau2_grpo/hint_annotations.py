"""Validated per-turn Qwen diagnosis, confidence and actionable hint."""
import re


def validate_annotation(annotation,supplied):
    if not isinstance(annotation,dict) or set(annotation)!={'diagnosis','confidence','hint'}:
        raise ValueError('Expected exactly diagnosis, confidence, hint')
    diagnosis=annotation['diagnosis'];hint=annotation['hint']
    if not isinstance(diagnosis,str) or not 0<len(diagnosis.split())<=40:
        raise ValueError('Diagnosis must be 1-40 words')
    if annotation['confidence'] not in ('high','medium','low'):
        raise ValueError('Confidence must be high, medium or low')
    if not isinstance(hint,str) or not 0<len(hint.split())<=24:
        raise ValueError('Hint must be 1-24 words')
    if not hint.startswith('Copy ' if supplied else 'Ask '):
        raise ValueError('Hint must match current Ask/Copy stage')
    if '<tool_call' in hint or (supplied and re.search(r'\bask\b|\bconfirm\b|\bprovide.*text',hint,re.I)):
        raise ValueError('Hint requests an inappropriate action')
    return annotation
