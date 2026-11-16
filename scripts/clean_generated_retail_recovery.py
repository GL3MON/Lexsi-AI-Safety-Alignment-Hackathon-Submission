"""Audit recovery examples without deleting the failed-call context."""
import json,shutil
from collections import Counter
from pathlib import Path
from clean_generated_retail_sft import strict_flags


def publish(root,rows):
    root.mkdir(parents=True,exist_ok=True);(root/'records').mkdir(exist_ok=True)
    for name,pool in [('all',rows),('sft',[r for r in rows if r['split']=='train']),('validation',[r for r in rows if r['split']=='validation'])]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pool))
    for row in rows:(root/'records'/(row['id'].rsplit('-',1)[-1]+'.json')).write_text(json.dumps(row,ensure_ascii=False))


def main():
    source=Path('outputs/retail-data/retail-sdg-v2-700/records');out=Path('outputs/retail-data/retail-sdg-v2-verified-recovery')
    good=[];excluded=[]
    for file in sorted(source.glob('*.json')):
        row=json.loads(file.read_text())
        if row['mode']!='recovery':continue
        flags=strict_flags(row,allow_recovery=True)
        if flags:excluded.append(dict(id=row['id'],source=str(file),reasons=flags))
        else:good.append(row)
    publish(out,good)
    report=dict(source_records=200,retained=len(good),train=sum(r['split']=='train' for r in good),
        validation=sum(r['split']=='validation' for r in good),excluded=len(excluded),
        reason_counts=dict(Counter(f['kind'] for r in excluded for f in r['reasons'])),
        failed_calls_retained_as_context_only=sum(len(r['masked_faults']) for r in good),
        messages_and_targets_unchanged=True,full_replay_and_labels_verified=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));(out/'excluded-manifest.json').write_text(json.dumps(excluded,indent=2))
    clean=[json.loads(l) for l in Path('outputs/retail-data/retail-sdg-v2-strict-clean/all.jsonl').open()]
    combined=Path('outputs/retail-data/retail-sdg-v2-curated');publish(combined,clean+good)
    (combined/'manifest.json').write_text(json.dumps(dict(clean=len(clean),recovery=len(good),total=len(clean)+len(good),
        train=sum(r['split']=='train' for r in clean+good),validation=sum(r['split']=='validation' for r in clean+good),
        original_splits_preserved=True,original_messages_and_target_masks_preserved=True),indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
