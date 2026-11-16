"""Recheck every clean subset label and preserve source task partitions."""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import read_records,encode_records,validate_task_partitions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/retail-data/retail-sdg-v2-strict-clean'))
    parser.add_argument('--allow-recovery',action='store_true')
    args=parser.parse_args();root=args.source
    train=read_records(root/'sft.jsonl');validation=read_records(root/'validation.jsonl')
    manifest=dict(partitions=dict(sft=[r['task_id'] for r in train],validation=[r['task_id'] for r in validation]))
    validate_task_partitions(train,validation,manifest)
    (root/'task-split-manifest.json').write_text(json.dumps(manifest,indent=2))
    tokenizer=AutoTokenizer.from_pretrained('outputs/rl-initialization/retail-sdg-v2-checkpoint-100')
    tokenizer.chat_template=Path('templates/retail_sdg_v2.jinja').read_text()
    counts=dict(records=0,total_tokens=0,supervised_tokens=0,max_tokens=0)
    for row,encoded,spans,error in encode_records(tokenizer,train+validation,8192,8):
        if error:raise RuntimeError(row['id']+': '+error)
        faults={f['message_index'] for f in row['masked_faults']}
        assert args.allow_recovery or (row['mode']=='clean' and not faults)
        assistant_indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
        assert set(row['loss_message_indices'])==set(assistant_indices)-faults
        selected=set()
        for index,(start,end) in zip(assistant_indices,spans,strict=True):
            if index in faults:assert all(value==-100 for value in encoded['labels'][start:end])
            else:
                assert encoded['labels'][start:end]==encoded['input_ids'][start:end]
                selected.update(range(start,end))
        assert all(value==-100 for i,value in enumerate(encoded['labels']) if i not in selected)
        counts['records']+=1;counts['total_tokens']+=len(encoded['input_ids']);counts['supervised_tokens']+=len(selected)
        counts['max_tokens']=max(counts['max_tokens'],len(encoded['input_ids']))
        if counts['records']%50==0:print('Verified clean labels:',counts['records'],flush=True)
    result=dict(**counts,all_passed=True,train_validation_group_overlap=0,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        train_sha256=hashlib.sha256((root/'sft.jsonl').read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256((root/'validation.jsonl').read_bytes()).hexdigest())
    (root/'label-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
