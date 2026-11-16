"""Package the validated Salesforce retail SFT run without credentials."""
import hashlib,json,tarfile
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
INPUT='GL3MON/tiny-aya-apigen-retail-sft-input'
RESULT='GL3MON/tiny-aya-apigen-retail-sft-results'

def main():
    data=ROOT/'outputs/retail-data/apigen-mt-retail-sft'
    manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['assistant_only_masks_verified'] and not manifest['train_validation_group_overlap']
    assert manifest['max_tokens']<=8192
    adapter=ROOT/'outputs/rl-glaive/run-55f242da3a/checkpoint-17/policy'
    artifact=HERE/'artifacts';artifact.mkdir(exist_ok=True)
    payload=artifact/'payload.tar.gz'
    with tarfile.open(payload,'w:gz',compresslevel=3) as archive:
        for tree,dest in [(ROOT/'src','project/src'),(ROOT/'templates','project/templates')]:
            for p in sorted(tree.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:archive.add(p,arcname=str(Path(dest)/p.relative_to(tree)))
        for name in ['pyproject.toml','README.md']:archive.add(ROOT/name,arcname='project/'+name)
        archive.add(ROOT/'scripts/probe_sft_batch_size.py',arcname='project/probe.py')
        archive.add(HERE/'config.json',arcname='project/sft-config.json')
        for name in ['train.jsonl','validation.jsonl','longest-tokenized.json','manifest.json']:
            archive.add(data/name,arcname='project/data/'+name)
        for name in ['adapter_config.json','adapter_model.safetensors','tokenizer_config.json','tokenizer.json','chat_template.jinja']:
            archive.add(adapter/name,arcname='project/initial-adapter/'+name)
    provenance=dict(dataset=manifest,initial_adapter=str(adapter),initial_adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),epochs=1,credentials_in_payload=False)
    probe=HERE/'validated-batch-probe.json'
    if probe.is_file():provenance['validated_batch_probe']=json.loads(probe.read_text())
    (artifact/'manifest.json').write_text(json.dumps(provenance,indent=2))
    api=HfApi()
    for repo in [INPUT,RESULT]:
        api.create_repo(repo,repo_type='dataset',private=True,exist_ok=True)
        assert api.repo_info(repo,repo_type='dataset').private
    for source,dest in [(payload,'payload.tar.gz'),(artifact/'manifest.json','manifest.json'),(HERE/'bootstrap.py','bootstrap.py')]:
        api.upload_file(path_or_fileobj=source,path_in_repo=dest,repo_id=INPUT,repo_type='dataset')
    print(json.dumps(dict(uploaded=True,payload_bytes=payload.stat().st_size,**manifest)),flush=True)

if __name__=='__main__':main()
