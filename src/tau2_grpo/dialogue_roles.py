"""Fail closed on clear speaker inversion before synthetic turns enter history."""
import re

def assert_speaker(role,text,history=()):
    if role not in ('assistant','user') or not isinstance(text,str) or not text.strip():raise ValueError('invalid_dialogue_turn')
    if role=='user' and re.search(r'<tool_(?:call|response)>',text):raise ValueError('customer_emitted_tool_markup')
    prose=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S)
    # Explicit quotations are data, not the speaker adopting the quoted identity.
    prose=re.sub(r'"[^"\n]*"|“[^”\n]*”','',prose)
    if role=='assistant':
        if re.search(r"\b(?:I(?:'ve| have)? (?:ordered|bought|purchased)|my (?:account|orders?|email|zip|address|payment|refund)|I (?:don't|do not|can't|cannot) remember (?:my|the specific order)|I(?:'d| would) like (?:a refund|to return|to cancel)|please (?:cancel|return|refund) my)\b",prose,re.I):
            raise ValueError('assistant_adopted_customer_perspective')
        normalized=' '.join(prose.lower().split())
        if len(normalized)>=40:
            for old in history:
                if old['role']=='user' and not old['content'].startswith('<tool_response>') and normalized==' '.join(old['content'].lower().split()):raise ValueError('assistant_copied_customer_turn')
    else:
        if re.search(r"\b(?:I(?:'ve| have)? (?:found|located|verified|authenticated) your (?:account|profile|orders?)|(?:I will|I'll|I can|let me) (?:process (?:your|the) (?:return|refund)|retrieve your|look up your|update your|cancel your|check your)|could you (?:please )?provide your (?:email|zip)|your user ID is)\b",prose,re.I):
            raise ValueError('customer_adopted_agent_perspective')
