"""Export a retail SFT checkpoint with its exact tokenizer/template; publish privately."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(filename):
    digest=hashlib.sha256()
    with Path(filename).open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--base-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo-id',required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    checkpoint=args.checkpoint.resolve();output=args.output.resolve()
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    adapter_config=json.loads((checkpoint/'adapter_config.json').read_text())
    api=HfApi()
    if args.repo_id.split('/')[0]!=api.whoami()['name']:raise ValueError('Expected your own HF account')
    api.create_repo(args.repo_id,repo_type='model',private=True,exist_ok=True)
    if not api.model_info(args.repo_id).private:raise RuntimeError('Expected a private model repository')
    output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    original_template=(checkpoint/'chat_template.jinja').read_text()
    if tokenizer.chat_template!=original_template:raise ValueError('Checkpoint template mismatch')
    examples=[{'role':'system','content':'You are a retail assistant.'},
              {'role':'user','content':'My email is ines.duarte+receipts@example.org.'},
              {'role':'assistant','content':'I will look up your account.\n<tool_call>{"name":"find_user_id_by_email","arguments":{"email":"ines.duarte+receipts@example.org"}}</tool_call>'},
              {'role':'tool','content':'{"name":"find_user_id_by_email","content":"ines_duarte_123"}'}]
    expected=tokenizer.apply_chat_template(examples,tokenize=True,add_generation_prompt=True,return_dict=False)
    tokenizer.save_pretrained(output)
    reloaded=AutoTokenizer.from_pretrained(output)
    if reloaded.apply_chat_template(examples,tokenize=True,add_generation_prompt=True,return_dict=False)!=expected:
        raise ValueError('Export changed prompt tokens')
    print('Exact SFT template and tool-response token round-trip verified.',flush=True)
    base=AutoModelForCausalLM.from_pretrained(str(args.base_path),dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False).merge_and_unload(safe_merge=True)
    model.config.eos_token_id=tokenizer.eos_token_id
    model.config.pad_token_id=tokenizer.pad_token_id
    model.generation_config.eos_token_id=tokenizer.eos_token_id
    model.generation_config.pad_token_id=tokenizer.pad_token_id
    model.save_pretrained(output,safe_serialization=True,max_shard_size='2GB')
    # Retain the source adapter separately, without optimizer state or training data.
    source_adapter=output/'source_adapter';source_adapter.mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):
        shutil.copyfile(checkpoint/name,source_adapter/name)
    tokenizer.save_pretrained(source_adapter)
    root=Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'templates/retail_sdg_v2_original_system.txt',output/'retail_system_prompt.txt')
    manifest=dict(source_checkpoint=str(checkpoint),step=state['global_step'],
        base_model=adapter_config['base_model_name_or_path'],base_revision=adapter_config.get('revision'),
        adapter_sha256=sha256(checkpoint/'adapter_model.safetensors'),
        template_sha256=hashlib.sha256(original_template.encode()).hexdigest(),
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
        dtype='bfloat16',merged=True,template_preserved=True,training_type='SFT',
        validation_loss=state.get('best_metric'))
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text(f'''---
library_name: transformers
pipeline_tag: text-generation
base_model: {manifest['base_model']}
tags:
- retail
- tool-use
- function-calling
- sft
- merged
---
# Tiny Aya Global retail SFT

Merged BF16 retail SFT checkpoint **{manifest['step']}** from the 700-example
retail synthetic-data run (630 training and 70 validation examples, two epochs).
The complete corpus contains 500 clean conversations, including calculator
examples, and 200 tool-error recovery conversations. Injected failed calls were
context-only training inputs, excluded from supervised targets.

This export preserves the checkpoint's exact tokenizer and Jinja chat template.
Assistant actions use `<tool_call>{{"name":"...","arguments":{{...}}}}</tool_call>`.
Store actual tool results with `role="tool"`; the template renders them in the
Aya tool-response format used during SFT. Use the included
`retail_system_prompt.txt` for the retail policy and catalog.

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

repo = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
policy = Path(hf_hub_download(repo, "retail_system_prompt.txt")).read_text()
messages = [{{"role": "system", "content": policy}},
            {{"role": "user", "content": "I need help with an order."}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
    return_dict=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    result = model.generate(**inputs, max_new_tokens=256, do_sample=False, logits_to_keep=1)
print(tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use Transformers for this checkpoint. The tested vLLM 0.17.1 configuration
produced incoherent long-prompt outputs; compatibility has not been established.
Validation loss does not establish successful end-to-end retail task execution.
Tool calls require application-side schema checks and execution.

`source_adapter/` preserves the unmerged source LoRA and tokenizer.
`export_manifest.json` records the exact checkpoint, hashes, and upstream revision.
Refer to the upstream model and dataset terms for applicable licenses.
''')
    manifest['files_sha256']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*')
                              if p.is_file() and p.name!='export_manifest.json'}
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    commit=api.upload_folder(repo_id=args.repo_id,folder_path=output,
        commit_message=f"Upload retail SFT checkpoint {manifest['step']} with unchanged SFT template")
    info=api.model_info(args.repo_id,files_metadata=True)
    if not info.private:raise RuntimeError('Repository privacy changed')
    uploaded={p.rfilename:p for p in info.siblings}
    for name,digest in manifest['files_sha256'].items():
        if name.endswith('.safetensors'):
            entry=uploaded[name]
            remote=entry.lfs['sha256'] if isinstance(entry.lfs,dict) else entry.lfs.sha256
            if remote!=digest:raise RuntimeError('Uploaded weights hash mismatch: '+name)
    print(json.dumps(dict(repo_url='https://huggingface.co/'+args.repo_id,private=True,
        step=manifest['step'],commit_url=str(commit),uploaded_weights_hashes_verified=True)),flush=True)


if __name__=='__main__':main()
