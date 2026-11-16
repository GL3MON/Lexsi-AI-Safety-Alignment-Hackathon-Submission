"""Merge a Tiny Aya SFT checkpoint, install its tool template, and publish privately."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    cfg = json.loads((checkpoint / 'adapter_config.json').read_text())
    api = HfApi()
    user = api.whoami()['name']
    if args.repo_id.split('/')[0] != user:
        raise ValueError('Repository must belong to the authenticated user')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.chat_template = (root / 'templates/tiny_aya_tool_calling.jinja').read_text()
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    messages = [{'role': 'system', 'content': 'Be helpful.'}, {'role': 'user', 'content': 'Weather in Paris?'}]
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    assert '<tools>' in rendered and 'get_weather' in rendered
    assert rendered.endswith('<|CHATBOT_TOKEN|><|START_RESPONSE|>')
    history = messages + [{'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}}]}, {'role': 'tool', 'content': '{"temperature_c":18}'}, {'role': 'tool', 'content': '{"condition":"sunny"}'}]
    tool_rendered = tokenizer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
    assert '<tool_response>{"temperature_c":18}</tool_response>' in tool_rendered
    assert '"arguments":{"city": "Paris"}' in tool_rendered
    tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hi'}], tokenize=True, add_generation_prompt=True)
    (args.output / 'rendered_tool_example.txt').write_text(tool_rendered)
    tokenizer.save_pretrained(args.output)
    reloaded = AutoTokenizer.from_pretrained(args.output)
    assert reloaded.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True) == rendered
    print('Template rendering and tokenizer round-trip passed.', flush=True)
    base = AutoModelForCausalLM.from_pretrained(cfg['base_model_name_or_path'], revision=cfg['revision'], dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).merge_and_unload(safe_merge=True)
    # Stop on both native response endings and the turn ending supervised during SFT.
    stops = model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, list) else [stops])
    stops.update([tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|END_RESPONSE|>')])
    model.generation_config.eos_token_id = sorted(x for x in stops if x is not None)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size='2GB')
    shutil.copy(root / 'templates/tool_calling_system_prompt.txt', args.output)
    shutil.copy(checkpoint / 'chat_template.jinja', args.output / 'original_chat_template.jinja')
    manifest = {'checkpoint': str(checkpoint), 'step': int(checkpoint.name.split('-')[-1]), 'base_model': cfg['base_model_name_or_path'], 'revision': cfg['revision'], 'adapter_sha256': hashlib.sha256((checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest(), 'dtype': 'bfloat16', 'template_render_checks_passed': True}
    (args.output / 'export_manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output / 'README.md').write_text(f'''---
library_name: transformers
pipeline_tag: text-generation
base_model: {cfg['base_model_name_or_path']}
tags:
- function-calling
- merged
- sft
---
# Tiny Aya function calling SFT
Merged BF16 Tiny Aya Global with LoRA checkpoint {manifest['step']} from a 3-epoch SFT run on a cleaned split of NousResearch/hermes-function-calling-v1. Refer to the upstream base model and dataset for their licenses and usage terms.

The bundled Jinja template preserves Aya's original preamble and turn tokens, adds explicit tool-call instructions and a worked example, accepts `tools=` in `apply_chat_template`, and maps standard tool messages to the user-role tool observations used during SFT. It also serializes assistant `tool_calls`. Supply tools as standard function schemas. Parse `<tool_call>{{"name":"...","arguments":{{...}}}}</tool_call>` blocks and execute them in your application.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
repo = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
tools = [{{"type":"function","function":{{"name":"get_weather","description":"Get current weather","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}},"required":["city"]}}}}}}]
messages = [{{"role":"user","content":"What is the weather in Paris?"}}]
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False, logits_to_keep=1)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Checkpoint {manifest['step']} scored 12/12 format/schema-valid calls, 9/12 exact reference calls, and 4/4 no-extra-call follow-ups on a fixed 16-case held-out offline evaluation using the ORIGINAL template. Those scores do not evaluate this modified prompt, semantic answer quality, or live tool execution. The new template passed rendering and tokenizer save/reload checks. Model outputs still require validation.

`original_chat_template.jinja` preserves the source template. `export_manifest.json` identifies the precise checkpoint and base revision. The trainer's final_adapter restored an earlier best-loss checkpoint; this export deliberately uses checkpoint {manifest['step']}.
''')
    # Create private before any weights are sent, and reject an existing public repository.
    repo = api.create_repo(args.repo_id, private=True, exist_ok=True)
    if not api.model_info(args.repo_id).private:
        raise RuntimeError('Refusing to upload to a public repository')
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=args.output, commit_message=f'Upload merged SFT checkpoint {manifest["step"]} and Aya tool-calling template')
    assert api.model_info(args.repo_id).private
    print(json.dumps({'repo_url': str(repo), 'private': True, 'commit_url': str(commit)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
