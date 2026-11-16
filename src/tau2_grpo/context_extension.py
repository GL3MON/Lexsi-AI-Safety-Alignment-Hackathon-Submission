"""Explicit AlignTune RoPE application for Tiny Aya's hybrid Cohere2 layers."""
import copy
import json
import math
from pathlib import Path


def extend_config(config,target_length,rope_type='yarn',factor=None,original_length=None,attention_factor=None):
    """Return a new config; preserve theta and local/global attention placement."""
    from tau2_grpo.aligntune_runtime import verify_aligntune
    from aligntune.core.long_context import RopeScalingConfig,RopeScalingApplier
    origin=verify_aligntune()
    if config.model_type!='cohere2':raise ValueError('This extension is validated only for Cohere2/Tiny Aya')
    original=original_length or config.max_position_embeddings
    if target_length<=original:raise ValueError('Target context must exceed the original context')
    ratio=target_length/original
    if factor is not None and not math.isclose(factor,ratio,rel_tol=1e-8):
        raise ValueError('RoPE factor must match target/original context lengths')
    if rope_type not in ('linear','dynamic','yarn','native'):raise ValueError('Unsupported context extension strategy')
    current=copy.deepcopy(config.rope_parameters)
    if not isinstance(current,dict) or current.get('rope_type','default')!='default':
        raise ValueError('Start from native unscaled RoPE; do not stack scaling transformations')
    if not isinstance(current.get('rope_theta'),(float,int)):raise ValueError('Missing native rope_theta')
    if not config.layer_types or 'full_attention' not in config.layer_types or 'sliding_attention' not in config.layer_types:
        raise ValueError('Expected native hybrid sliding/global attention')
    extended=copy.deepcopy(config)
    if rope_type!='native':
        scaling=RopeScalingApplier.build_rope_config(RopeScalingConfig(
            type=rope_type,factor=ratio,original_max_position=original,target_max_position=target_length))
        scaling.pop('type',None)
        extended.rope_parameters={**current,**scaling}
        if attention_factor is not None:extended.rope_parameters['attention_factor']=attention_factor
    elif attention_factor is not None:raise ValueError('Attention factor requires a scaled RoPE strategy')
    extended.max_position_embeddings=target_length
    if extended.sliding_window!=config.sliding_window or extended.layer_types!=config.layer_types:
        raise RuntimeError('Context extension changed the native attention architecture')
    # Construct actual Transformers rotary math now, so a silent skip is fatal.
    from transformers.models.cohere2.modeling_cohere2 import Cohere2RotaryEmbedding
    rotary=Cohere2RotaryEmbedding(extended)
    expected='default' if rope_type=='native' else rope_type
    if rotary.rope_type!=expected:raise RuntimeError('Requested RoPE strategy was not applied')
    manifest=dict(version=1,strategy=rope_type,original_max_position_embeddings=original,
        target_max_position_embeddings=target_length,factor=ratio,
        rope_parameters=copy.deepcopy(extended.rope_parameters),sliding_window=extended.sliding_window,
        layer_types=extended.layer_types,rotary_applied_to='sliding_attention only; global layers retain NoPE',
        quality_validated=False,**origin)
    return extended,manifest


def save_extension(directory,config,manifest):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    config.save_pretrained(directory)
    (directory/'context_extension.json').write_text(json.dumps(manifest,indent=2)+'\n')


def checkpoint_callback(config,manifest):
    from transformers import TrainerCallback
    class SaveContextExtension(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            save_extension(Path(args.output_dir)/f'checkpoint-{state.global_step}',config,manifest)
    return SaveContextExtension()
