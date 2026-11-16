"""Canonical tool roles; Aya Jinja owns their model-specific rendering."""
import json


def canonical_tool_messages(messages):
    result=[]
    for message in messages:
        m=dict(message)
        if m['role']=='user' and m['content'].startswith('<tool_response>') and m['content'].endswith('</tool_response>'):
            # Preserve the exact serialized envelope seen by the trained template.
            payload=m['content'][len('<tool_response>'):-len('</tool_response>')]
            observation=json.loads(payload)
            if not isinstance(observation,dict) or not isinstance(observation.get('name'),str) or 'content' not in observation:
                raise ValueError('Invalid tool observation envelope')
            m.update(role='tool',name=observation['name'],content=payload)
        result.append(m)
    return result


def template_supports_tool_roles(tokenizer):
    template=getattr(tokenizer,'chat_template',None)
    return isinstance(template,str) and 'tool_response' in template and ("== 'tool'" in template or '== "tool"' in template)
