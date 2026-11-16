"""Causal, frozen-summary contexts. Replay uses saved tokens, never this controller."""
import copy
import hashlib
import json
from pathlib import Path

from tau2_grpo.prepare_function_calling import model_messages
from tau2_grpo.retail_memory import summary_errors, controller_history

ROOT = Path(__file__).resolve().parents[2]


def normalize_provenance(memory, raw):
    """Resolve call citations to their actual adjacent result, then validate facts.

    This does not search for a convenient observation or change item/order values.
    Authentication is controller-owned and requires a successful lookup result.
    """
    memory = copy.deepcopy(memory)
    corrections = []
    def visit(value):
        if isinstance(value, list):
            for child in value: visit(child)
        elif isinstance(value, dict):
            if isinstance(value.get('source_ids'), list):
                ids = []
                for i in value['source_ids']:
                    mapped = i
                    if type(i) is int and 0 < i < len(raw)-1 and raw[i]['role'] == 'assistant' and raw[i+1]['role'] == 'tool':
                        calls = raw[i].get('tool_calls') or []
                        if not calls:
                            from tau2_grpo.prepare_function_calling import parse_calls
                            try:calls=parse_calls(raw[i].get('content',''))
                            except ValueError:calls=[]
                        if len(calls) == 1:
                            name = calls[0].get('function', calls[0]).get('name')
                            result = json.loads(raw[i+1]['content'])
                            if name == result.get('name'):
                                mapped = i+1
                                corrections.append(dict(kind='call_to_paired_result', before=i, after=mapped))
                    ids.append(mapped)
                value['source_ids'] = list(dict.fromkeys(ids))
            for child in value.values(): visit(child)
    visit(memory)
    customer = memory.get('customer')
    if isinstance(customer, dict):
        proof = []
        for i, message in enumerate(raw):
            if message['role'] == 'tool':
                result = json.loads(message['content'])
                if result.get('name') in {'find_user_id_by_email', 'find_user_id_by_name_zip'} and customer.get('user_id') is not None and result.get('content') == customer['user_id']:
                    proof.append(i)
        verified = bool(proof)
        if customer.get('authenticated') != verified:
            corrections.append(dict(kind='controller_authentication', value=verified, source_ids=proof))
        customer['authenticated'] = verified
        if proof: customer['source_ids'] = sorted(set(customer.get('source_ids', [])) | set(proof))
    return memory, corrections


def student_messages(raw):
    """Match SFT's compact JSON and Aya tool-result rendering."""
    import re
    result = []
    for m in model_messages(raw):
        content = m['content']
        for tag in ('tools', 'tool_call', 'tool_response'):
            def compact(match):
                try:
                    return '<'+tag+'>'+json.dumps(json.loads(match[1]), separators=(',', ':'), ensure_ascii=False)+'</'+tag+'>'
                except ValueError:
                    return match[0]  # Malformed policy actions remain recoverable observations.
            content = re.sub('<'+tag+r'>\s*(.*?)\s*</'+tag+'>',
                             compact,
                             content, flags=re.S)
        result.append({'role': m['role'], 'content': content})
    return result


class ContextError(RuntimeError):
    pass


class FrozenSummarizer:
    def __init__(self, client, model, tokenizer, context_limit=16384, max_tokens=2000):
        self.client, self.model, self.tokenizer = client, model, tokenizer
        self.context_limit, self.max_tokens = context_limit, max_tokens
        self.prompt = (ROOT/'templates/retail_memory_summarizer.txt').read_text()
        self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()
        self.attempts = []

    def __call__(self, raw, tail_start, seed, attempt, prior=None):
        numbered = [{'message_id': i, **m} for i, m in enumerate(raw) if i]
        payload = dict(policy=raw[0]['content'], original_request=raw[1]['content'],
                       history_to_summarize=numbered[:tail_start-1],
                       recent_messages_kept_verbatim=numbered[tail_start-1:],
                       recent_tail_message_ids=list(range(tail_start, len(raw))))
        if prior:
            # Bound repeated summarizer inputs without renumbering original citations.
            payload['previous_validated_memory'] = prior['memory']
            payload['previous_recent_exchange'] = [m for m in numbered if prior['tail_start'] <= m['message_id'] < prior['raw_message_count']]
            payload['history_to_summarize'] = [m for m in numbered if prior['raw_message_count'] <= m['message_id'] < tail_start]
        request = [{'role': 'system', 'content': self.prompt},
                   {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        if attempt and self.attempts:
            failures=self.attempts[-1].get('validation_errors') or []
            if failures:
                request[0]['content']+='\nThe previous candidate failed factual validation: '+json.dumps(failures[:12])+'. Recheck the cited observations and return a replacement JSON summary. Unknown fields must be null or omitted; never invent replacements or weaken evidence.'
        count = len(self.tokenizer.apply_chat_template(request, tokenize=True, return_dict=False,
                     add_generation_prompt=True, enable_thinking=False))
        if count+self.max_tokens > self.context_limit:
            raise ContextError('Frozen summarizer input exceeds its supported context; raw episode retained')
        response = self.client.chat.completions.create(
            model=self.model, messages=request, temperature=0 if attempt == 0 else 0.2,
            seed=seed+attempt, max_tokens=self.max_tokens, response_format={'type': 'json_object'},
            extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        audit = dict(request=request, response=response.model_dump(), prompt_hash=self.prompt_hash,
                     model=self.model, seed=seed+attempt)
        self.attempts.append(audit)
        if response.choices[0].finish_reason == 'length':
            raise ContextError('Truncated summary')
        memory = json.loads(response.choices[0].message.content or '')
        if not isinstance(memory, dict):
            raise ContextError('Summary is not an object')
        return memory, audit


class ContextController:
    def __init__(self, tokenizer, summarizer, context_limit=8192, response_tokens=512, prompt_budget=None):
        self.tokenizer, self.summarizer = tokenizer, summarizer
        self.context_limit, self.response_tokens = context_limit, response_tokens
        self.budget = prompt_budget or context_limit-response_tokens
        if not 0 < self.budget <= context_limit-response_tokens:
            raise ValueError('Prompt budget must preserve the full response reserve')
        self.active = None
        self.next_raw = 0
        self.summaries = []

    def render(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if ids != self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True):
            raise ContextError('Rendered text/token IDs disagree')
        return dict(messages=copy.deepcopy(messages), rendered_prompt=prompt, prompt_ids=ids)

    def prepare(self, raw, seed, guidance=None):
        view = student_messages(raw)
        current = copy.deepcopy(self.active) + view[self.next_raw:] if self.active else view
        result = self.render(current)
        if len(result['prompt_ids']) > self.budget:
            # The last complete assistant+incoming exchange remains verbatim.
            # Raw history is never replaced; repeated compactions remain causal.
            previous = [i for i, m in enumerate(raw) if m['role'] == 'assistant']
            if not previous:
                raise ContextError('Policy and opening customer message do not fit')
            tail_start = previous[-1]
            failures = []
            for attempt in range(3):
                try:
                    memory, audit = self.summarizer(copy.deepcopy(raw), tail_start, seed, attempt,
                                                   prior=self.summaries[-1] if self.summaries else None)
                    memory, audit['controller_corrections'] = normalize_provenance(memory, raw)
                    memory.pop('confirmation', None)
                    memory['observed_history'] = controller_history(raw)
                    memory['recent_exchange'] = dict(source_ids=list(range(tail_start, len(raw))), owner='controller',
                        instruction='Read the following original turns; they supersede older memory. Do not infer consent or execution from this pointer.')
                    errors = summary_errors(memory, raw)
                    audit['validation_errors'] = errors
                    if errors:
                        raise ContextError('; '.join(errors))
                    text = 'Original customer request:\n'+raw[1]['content']+'\n\nPrior working memory (not a new customer instruction):\n<memory>'+json.dumps(memory, separators=(',', ':'), ensure_ascii=False)+'</memory>'
                    current = [view[0], {'role': 'user', 'content': text}] + view[tail_start:]
                    candidate = self.render(current)
                    if len(candidate['prompt_ids']) > self.budget:
                        raise ContextError('Summary plus intact recent exchange exceeds prompt budget')
                    audit.update(memory=memory, tail_start=tail_start, raw_message_count=len(raw), prompt_tokens=len(candidate['prompt_ids']))
                    self.summaries.append(audit)
                    result = candidate
                    break
                except (ContextError, ValueError, TypeError, KeyError) as error:
                    failures.append(str(error))
            else:
                raise ContextError('Summary preparation failed: '+repr(failures))
        # Guidance is privileged, single-turn context. Keep it out of the raw
        # transcript and controller cache so it cannot leak into later turns.
        prepared=result
        if guidance:
            guided=copy.deepcopy(result['messages'])
            target=next((i for i in range(len(guided)-1,-1,-1) if guided[i]['role']=='user'),None)
            note='\n\nGuidance for your next response: '+str(guidance)+'\nFollow this guidance for the next action.'
            if target is None:
                guided.append({'role':'user','content':note.lstrip()})
            else:
                guided[target]['content']+=note
            prepared=self.render(guided)
            if len(prepared['prompt_ids'])>self.budget:
                raise ContextError('Privileged guidance exceeds the actor prompt budget')
        self.active, self.next_raw = result['messages'], len(raw)
        assert len(prepared['prompt_ids'])+self.response_tokens <= self.context_limit
        return prepared
