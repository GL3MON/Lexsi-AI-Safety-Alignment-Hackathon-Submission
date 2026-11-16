"""Narrow, version-checked fix for Cohere's tied embedding with vLLM LoRA."""
import ast
import importlib.metadata
import importlib.util
from pathlib import Path

OLD = 'is_not_lora = hasattr(self.model.embed_tokens, "weight")'
NEW = 'is_not_lora = not hasattr(self.model.embed_tokens, "base_layer")'


def patch_cohere_lora(source):
    # EmbeddingWithLoRA forwards .weight, so that attribute cannot identify a
    # bare embedding. The existing else branch already unwraps .base_layer.
    if NEW in source:
        return source
    if source.count(OLD) != 1:
        raise RuntimeError('Unexpected vLLM Cohere code; refusing an unverified patch')
    result = source.replace(OLD, NEW)
    ast.parse(result)
    return result


def main():
    if importlib.metadata.version('vllm') != '0.17.1':
        raise RuntimeError('This compatibility patch is verified only for vLLM 0.17.1')
    spec = importlib.util.find_spec('vllm')
    filename = Path(spec.origin).parent / 'model_executor/models/commandr.py'
    original = filename.read_text()
    updated = patch_cohere_lora(original)
    if updated != original:
        filename.with_suffix('.py.before-tied-lora-fix').write_text(original)
        filename.write_text(updated)
    print('Applied verified Cohere tied-embedding LoRA dispatch fix', flush=True)


if __name__ == '__main__':
    main()
