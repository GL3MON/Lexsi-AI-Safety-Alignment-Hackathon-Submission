"""Canonical tool prompts: one catalog, no literal delimiter examples."""
import json
import re
from pathlib import Path
from tau2_grpo.prepare_function_calling import tool_validators


def system_prompt(tools):
    tool_validators(tools)
    rules=(Path(__file__).resolve().parents[2]/'templates/glaive_decision_rules.txt').read_text().strip()
    return ('You are a function-calling assistant. Use only the available functions. '
            'Respond naturally when asking for information. When calling a function, '
            'output its name and arguments as one JSON object inside a tool_call XML block.\n\n'+
            rules+'\n\nAvailable tools:\n<tools>\n'+json.dumps(tools,ensure_ascii=False)+'\n</tools>')


def check_system(text,tools):
    if text.count('<tools>')!=1 or text.count('</tools>')!=1:raise ValueError('Expected exactly one tool catalog')
    match=re.search(r'<tools>(.*?)</tools>',text,re.S)
    if json.loads(match[1])!=tools:raise ValueError('Embedded tool catalog mismatch')


def clean_tokenizer_template(tokenizer):
    """Remove an unmatched documentation tag, retaining all control tokens."""
    template=tokenizer.chat_template
    if not isinstance(template,str):raise ValueError('Expected a string chat template')
    tokenizer.chat_template=template.replace('Use the functions listed in <tools> when',
                                             'Use the functions in the available tool catalog when')
    return tokenizer.chat_template
