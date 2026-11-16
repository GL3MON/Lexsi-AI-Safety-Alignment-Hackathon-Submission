"""Keep APIGen-MT retail conversations without changing their original schema."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

REPO='Salesforce/APIGen-MT-5k'
REVISION='abc4a517d67c541f85f6470cbd8fd3186b36830e'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/retail-data/apigen-mt-retail'))
    parser.add_argument('--revision',default=REVISION)
    args=parser.parse_args()
    source=args.input
    if source is None:
        from huggingface_hub import hf_hub_download
        source=Path(hf_hub_download(REPO,'apigen-mt_5k.json',repo_type='dataset',revision=args.revision))
    raw=source.read_bytes();rows=json.loads(raw)
    if not isinstance(rows,list):raise ValueError('Expected a JSON array of source records')
    selected=[];indices=[];domains=Counter();systems=Counter()
    for index,row in enumerate(rows):
        system=row.get('system')
        if not isinstance(system,str):raise ValueError(f'Missing string system prompt at row {index}')
        retail=bool(re.search(r'\bretail\b',system,re.I))
        airline=bool(re.search(r'\bairline\b',system,re.I))
        if retail and airline:raise ValueError(f'Ambiguous domain at row {index}')
        domain='retail' if retail else 'airline' if airline else 'unmatched'
        domains[domain]+=1
        if retail:
            tools=json.loads(row['tools']) if isinstance(row['tools'],str) else row['tools']
            names={t.get('name',t.get('function',{}).get('name')) for t in tools}
            if not {'find_user_id_by_email','cancel_pending_order'}<=names:
                raise ValueError(f'Retail policy/tool catalog disagreement at row {index}')
            selected.append(row);indices.append(index);systems[system]+=1
    if not selected:raise ValueError('No retail records matched')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/'retail.jsonl'
    with output.open('w') as stream:
        for row in selected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Verify exact semantic preservation, including tool schemas and role labels.
    saved=[json.loads(line) for line in output.open()]
    if saved!=selected:raise RuntimeError('Saved rows differ from source')
    manifest=dict(source_repo=REPO,source_revision=args.revision,source_file='apigen-mt_5k.json',
        source_sha256=hashlib.sha256(raw).hexdigest(),source_records=len(rows),
        source_fields=sorted({key for row in rows for key in row}),
        domain_metadata_present=any('metadata' in row or 'domain' in row for row in rows),
        selection={'field':'system','pattern':r'\bretail\b','case_sensitive':False},
        domain_counts=dict(domains),retail_records=len(selected),
        selected_source_row_indices=indices,unique_retail_system_prompts=len(systems),
        conversation_role_counts=dict(Counter(m['from'] for row in selected for m in row['conversations'])),
        output_file=output.name,output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        original_fields_preserved=True,token_length_filter_applied=False,train_validation_split_applied=False)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    labels={'human':'USER','gpt':'ASSISTANT','function_call':'ASSISTANT TOOL CALL','observation':'TOOL RESPONSE'}
    preview='SYSTEM:\n'+selected[0]['system']+'\n\n'+'\n\n'.join(
        labels.get(m['from'],m['from'].upper())+':\n'+m['value'] for m in selected[0]['conversations'])
    (args.output_dir/'sample-0000.txt').write_text(preview+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='selected_source_row_indices'},indent=2))


if __name__=='__main__':main()
