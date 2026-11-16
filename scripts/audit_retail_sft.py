"""Audit finalized retail SFT token budgets, masks, tokenizer parity, and target coverage.

Run with .venv/bin/python scripts/audit_retail_sft.py. Writes training-audit.json.
"""
import importlib.util, json, collections, hashlib
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records, tokenize_assistant_only, validate_task_partitions
root = Path(__file__).resolve().parents[1]
base = root / 'outputs/retail-data/retail-sft-final'
spec = importlib.util.spec_from_file_location('finalizer', root / 'scripts/finalize_retail_sft.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
train = read_records(base / 'sft.jsonl')
val = read_records(base / 'validation.jsonl')
validate_task_partitions(train, val, json.loads((root / 'outputs/retail-data/split_manifest.json').read_text()))
tok = AutoTokenizer.from_pretrained('GL3MON/tiny-aya-global-function-calling-sft', revision='df7ccd20bf3e5234bfbe7507821a5ca6ce814a2a', local_files_only=True)
local = AutoTokenizer.from_pretrained(root / 'outputs/hf-export/tiny-aya-global-function-calling-sft-525', local_files_only=True)
coverage = collections.defaultdict(list)
originals = {}
report = {'splits': {}, 'tokenizer_parity': True, 'tool_call_masking': {}}
for split, rows in [('train', train), ('validation', val)]:
    lengths = []
    targets = 0
    repeated = 0
    tool_masks = dict(supervised_tool_call_messages=0, supervised_tool_calls=0,
                      masked_repeated_tool_call_messages=0, masked_tool_response_messages=0)
    for row in rows:
        messages = row['messages']
        encoded, spans = tokenize_assistant_only(tok, messages, 8192, row.get('loss_message_indices'))
        assert encoded['input_ids'] == local.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False)
        selected = set(row['loss_message_indices'])
        expected = [-100] * len(encoded['labels'])
        for i, (a, b) in zip((i for i, m in enumerate(messages) if m['role'] == 'assistant'), spans):
            if i in selected:
                expected[a:b] = encoded['input_ids'][a:b]
            else:
                repeated += 1
        assert encoded['labels'] == expected
        for i, message in enumerate(messages):
            if message['role'] == 'assistant' and '<tool_call>' in message['content']:
                if i in selected:
                    tool_masks['supervised_tool_call_messages'] += 1
                    tool_masks['supervised_tool_calls'] += message['content'].count('<tool_call>')
                else:
                    tool_masks['masked_repeated_tool_call_messages'] += 1
            if message['role'] == 'tool' or (message['role'] == 'user' and '<tool_response>' in message['content']):
                assert i not in selected
                tool_masks['masked_tool_response_messages'] += 1
        source = Path(row['source_episode'])
        if str(source) not in originals:
            candidate = source.parents[2] / 'exports-all-generations' / source.parent.name / (row['partition'] + '-candidates.jsonl')
            full = next((r for r in mod.read_rows(candidate) if r['source_episode'] == str(source)))
            originals[str(source)] = mod.canonical(full['messages'])
        original = originals[str(source)]
        for i in row['loss_message_indices']:
            original_i = row['source_message_indices'][i]
            assert messages[i] == original[original_i]
            coverage[str(source)].append(original_i)
        assert [row['source_message_indices'][i] for i in row['loss_message_indices']] == row['source_target_indices']
        lengths.append(len(encoded['input_ids']))
        targets += sum((x != -100 for x in encoded['labels']))
    report['splits'][split] = {'examples': len(rows), 'max_tokens': max(lengths), 'supervised_tokens': targets, 'masked_repeated_assistant_messages': repeated}
    assert tool_masks['supervised_tool_calls'] > 0
    report['tool_call_masking'][split] = tool_masks
for source, indices in coverage.items():
    expected = [i for i, m in enumerate(originals[source]) if m['role'] == 'assistant']
    assert sorted(indices) == expected and len(indices) == len(set(indices))
report.update(selected_dataset_ready=True, source_trajectories=len(coverage), assistant_targets_unchanged_and_once_only=True, loss_masks_verified=True, train_validation_tasks_disjoint=True, excluded_failed_source_trajectories=json.loads((base / 'report.json').read_text())['failed_trajectories'], sha256={f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in ['sft.jsonl', 'validation.jsonl']})
(base / 'training-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
