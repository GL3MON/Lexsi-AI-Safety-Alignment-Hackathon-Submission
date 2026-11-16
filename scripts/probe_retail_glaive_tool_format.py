"""Audit saved tool exposure and probe Glaive-style framing on the same opening."""
import json
import re
from pathlib import Path


def glaive_framed_system(system):
    """Preserve the full retail policy and catalog with Glaive-style framing."""
    blocks = re.findall(r'<tools>(.*?)</tools>', system, re.S)
    if len(blocks) != 1:
        raise ValueError('Expected exactly one tool catalog')
    tools = json.loads(blocks[0])
    if len(tools) != len({t['function']['name'] for t in tools}):
        raise ValueError('Duplicate tool names')
    remaining = re.sub(r'<tools>.*?</tools>', '', system, flags=re.S).strip()
    schema = {'title': 'FunctionCall', 'type': 'object',
              'properties': {'arguments': {'title': 'Arguments', 'type': 'object'},
                             'name': {'title': 'Name', 'type': 'string'}},
              'required': ['arguments', 'name']}
    return (
        "You are a function calling AI model. You are provided with function signatures below. "
        "You may call functions to assist with the user query. Don't make assumptions about "
        "what values to plug into functions. Here are the available tools:\n<tools>\n"
        + json.dumps(tools, ensure_ascii=False) + '\n</tools>\n'
        + 'Use the following pydantic model JSON schema for each tool call: '
        + json.dumps(schema) + '\nFor each function call return a JSON object with function '
        + 'name and arguments within tool_call XML tags.\n\n' + remaining)


def main():
    import torch
    from tau2_grpo.rl_config import RetailRLConfig
    from tau2_grpo.rl_context import student_messages
    from tau2_grpo.rl_policy import LocalPolicy

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path('outputs/retail-eval')
    out = root / 'task63-glaive-format-probe'
    out.mkdir(parents=True, exist_ok=True)
    names = [
        ('pre_sft', 'pre-sdg-v2-glaive-rl17-task63-prompt-v2'),
        ('retail_sft50', 'sdg-v2-all700-checkpoint50-task63-prompt-v2'),
    ]
    seed = 20261113
    records = {}
    for label, folder in names:
        episode = json.loads((root / folder / 'episode.json').read_text())
        action = episode['actions'][0]
        messages = [dict(m) for m in action['messages']]
        system = messages[0]['content']
        tools = json.loads(re.findall(r'<tools>(.*?)</tools>', system, re.S)[0])
        messages[0]['content'] = glaive_framed_system(system)
        cfg = RetailRLConfig(initial_adapter=str(root / folder / 'checkpoint_snapshot'),
                             device='cpu', temperature=.8, top_p=1.)
        policy = LocalPolicy(cfg)
        policy.tokenizer.chat_template = Path('templates/retail_glaive_eval.jinja').read_text()
        rendered = policy.tokenizer.apply_chat_template(student_messages(messages),
                                                        tokenize=False, add_generation_prompt=True)
        encoded = policy.tokenizer.encode(rendered, add_special_tokens=False)
        catalog = re.findall(r'<tools>(.*?)</tools>', rendered, re.S)
        assert len(catalog) == 1 and json.loads(catalog[0]) == tools
        assert 'get_weather' not in rendered
        assert len(encoded) + 512 <= cfg.context_limit
        (out / f'{label}.rendered-prompt.txt').write_text(rendered)
        print(label, 'generating', len(encoded), 'prompt tokens', flush=True)
        result = policy.generate({'prompt_ids': encoded}, 512, seed)
        records[label] = dict(original=action['text'], revised=result['text'],
                              seed=seed, temperature=.8, top_p=1., tool_count=len(tools),
                              prompt_tokens=len(encoded), ended_with_eos=result['ended_with_eos'],
                              note='First-turn prompt probe, no environment execution or task reward.')
        (out / 'results.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
        text = '\n\n'.join(
            f"MODEL: {name}\nUSER: {messages[-1]['content']}\n\nORIGINAL FRAMING:\n{row['original']}"
            f"\n\nGLAIVE-STYLE FRAMING:\n{row['revised']}" for name, row in records.items())
        (out / 'comparison.txt').write_text(text + '\n')
        print(label, result['text'], flush=True)
        del policy
        import gc
        gc.collect()


if __name__ == '__main__':
    main()
