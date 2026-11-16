"""Greedy exact-prefix diagnostic: observed failure versus a training call target."""
import argparse
import json
import time
from pathlib import Path


def main():
    import torch
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    from tau2_grpo.rl_config import RetailRLConfig
    from tau2_grpo.rl_policy import LocalPolicy
    from tau2_grpo.inference_head_offload import enable_bounded_cuda_head
    from tau2_grpo.prepare_function_calling import parse_calls

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise ValueError('Probe output already exists')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cfg = RetailRLConfig(initial_adapter=str(args.checkpoint.resolve()), device='cpu', temperature=0)
    policy = LocalPolicy(cfg)
    policy.tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    policy.tokenizer.chat_template = (args.checkpoint / 'chat_template.jinja').read_text()
    saved = load_file(str(args.checkpoint / 'adapter_model.safetensors'))
    loaded = get_peft_model_state_dict(policy.model, adapter_name='policy')
    mismatches = [name for name, value in saved.items()
                  if name not in loaded or not torch.equal(value, loaded[name].cpu())]
    if mismatches:
        raise RuntimeError(f'Loaded adapter mismatch: {mismatches[:5]}')
    (out / 'adapter-audit.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint), 'active_adapters': policy.model.active_adapters,
        'saved_tensors': len(saved), 'mismatches': mismatches}, indent=2))
    del saved, loaded
    try:
        placement = enable_bounded_cuda_head(policy, 256)
    except RuntimeError as error:
        if 'Insufficient free memory' not in str(error):
            raise
        placement = {'mode': 'cpu_fallback', 'reason': str(error)}
    (out / 'gpu-placement.json').write_text(json.dumps(placement, indent=2))
    previous = json.loads(Path(
        'outputs/retail-eval/task63-original-sft-prompt-greedy/sft50/episode.json').read_text())
    action = previous['actions'][1]
    assert policy.tokenizer.encode(action['rendered_prompt'], add_special_tokens=False) == action['prompt_ids']
    record = json.loads(Path('outputs/retail-data/retail-sdg-v2-700/sft.jsonl').open().readline())
    first = next(i for i, message in enumerate(record['messages']) if message['role'] == 'assistant')
    training_messages = record['messages'][:first]
    training_ids = policy.tokenizer.apply_chat_template(
        training_messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    cases = [
        ('chen_email_exact_failed_prefix', action['messages'], action['prompt_ids'],
         'find_user_id_by_email', action['text'], None),
        ('exact_training_prefix', training_messages, training_ids,
         'find_user_id_by_email', None, record['messages'][first]['content']),
    ]
    results = []
    for name, messages, ids, expected_name, previous_text, target in cases:
        (out / 'progress.json').write_text(json.dumps({'stage': 'generating', 'case': name}))
        start = time.monotonic()
        result = policy.generate({'prompt_ids': ids}, 256, previous['seed'])
        try:
            calls = parse_calls(result['text'])
            parse_error = None
        except ValueError as error:
            calls, parse_error = [], str(error)
        row = {'case': name, 'messages': messages, 'text': result['text'],
               'calls': calls, 'parse_error': parse_error,
               'expected_tool': expected_name, 'checkpoint50_text': previous_text,
               'training_target': target, 'ended_with_eos': result['ended_with_eos'],
               'generation_seconds': time.monotonic() - start,
               'note': 'Prefix-only diagnostic; generated tools are not executed.'}
        results.append(row)
        (out / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
        paragraphs = [f'CHECKPOINT: {args.checkpoint}\nGREEDY PREFIX PROBES; no environment execution.']
        for item in results:
            paragraphs.append('CASE: ' + item['case'])
            paragraphs.extend(m['role'].upper() + ':\n' + m['content']
                              for m in item['messages'] if m['role'] != 'system')
            if item['checkpoint50_text'] is not None:
                paragraphs.append('CHECKPOINT 50 OUTPUT:\n' + item['checkpoint50_text'])
            paragraphs.append('NEW CHECKPOINT OUTPUT:\n' + item['text'])
            if item['training_target'] is not None:
                paragraphs.append('TRAINING TARGET (not supplied during generation):\n' + item['training_target'])
        (out / 'conversations.txt').write_text('\n\n'.join(paragraphs) + '\n')
        print('RESULT', json.dumps(row, ensure_ascii=False), flush=True)
    (out / 'progress.json').write_text(json.dumps({'stage': 'complete', 'cases': len(results)}))


if __name__ == '__main__':
    main()
