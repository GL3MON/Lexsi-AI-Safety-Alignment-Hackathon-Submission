"""Exact sampled-token probabilities and memory-bounded, differentiable replay."""
from contextlib import contextmanager


def disable_dropout(model):
    import torch
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.0
    for name in ('attention_dropout','hidden_dropout','attention_probs_dropout_prob'):
        if getattr(model.config,name,0):
            raise ValueError('Turn-aware replay requires zero '+name)


def action_logprobs(model, prompt_ids, completion_ids, temperature=1.0, chunk_size=16):
    """Only generated positions get logits/loss; never materialize prompt logits."""
    import torch
    from torch.utils.checkpoint import checkpoint
    if not prompt_ids or not completion_ids: raise ValueError('Empty prompt/completion')
    device=next(model.parameters()).device
    ids=torch.tensor([prompt_ids+completion_ids],device=device)
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    hidden=base.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
    selected=hidden[0,len(prompt_ids)-1:-1]
    target=ids[0,len(prompt_ids):]
    def project(h,t):
        # Cohere2 scales in native dtype before generation converts to FP32.
        logits=(base.lm_head(h)*getattr(base,'logit_scale',1.0)).float()/temperature
        return logits.gather(-1,t[:,None]).squeeze(-1)-torch.logsumexp(logits,dim=-1)
    parts=[]
    for start in range(0,len(target),chunk_size):
        h,t=selected[start:start+chunk_size],target[start:start+chunk_size]
        parts.append(checkpoint(project,h,t,use_reentrant=False) if torch.is_grad_enabled() else project(h,t))
    return torch.cat(parts)


def action_logprobs_batch(model, actions, pad_token_id, temperature=1., chunk_size=16):
    """Left-padded replay with logical positions and per-action generated-token targets."""
    import torch
    from torch.utils.checkpoint import checkpoint
    if not actions:raise ValueError('Empty replay batch')
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    device=next(model.parameters()).device
    lengths=[len(a['prompt_ids'])+len(a['completion_ids']) for a in actions]
    width=max(lengths)
    ids=torch.full((len(actions),width),pad_token_id,device=device,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for row,(action,length) in enumerate(zip(actions,lengths,strict=True)):
        if not action['prompt_ids'] or not action['completion_ids']:raise ValueError('Empty action')
        ids[row,-length:]=torch.tensor(action['prompt_ids']+action['completion_ids'],device=device)
        mask[row,-length:]=1
    positions=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
    hidden=base.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state
    def project(h,target):
        logits=(base.lm_head(h)*getattr(base,'logit_scale',1.)).float()/temperature
        return logits.gather(-1,target[:,None]).squeeze(-1)-torch.logsumexp(logits,dim=-1)
    results=[]
    for row,(action,length) in enumerate(zip(actions,lengths,strict=True)):
        start=width-length+len(action['prompt_ids'])-1
        selected=hidden[row,start:width-1]
        target=torch.tensor(action['completion_ids'],device=device)
        pieces=[]
        for offset in range(0,len(target),chunk_size):
            h,t=selected[offset:offset+chunk_size],target[offset:offset+chunk_size]
            pieces.append(checkpoint(project,h,t,use_reentrant=False) if torch.is_grad_enabled() else project(h,t))
        results.append(torch.cat(pieces))
    return results


class LocalPolicy:
    def __init__(self,cfg,adapter=True,trainable=False):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel
        from tau2_grpo.rl_config import path
        self.cfg=cfg; self.version=0; self.has_reference=False
        model_path=str(path(cfg.model_id)) if path(cfg.model_id).exists() else cfg.model_id
        # The adapter's tokenizer/template is the SFT contract, including EOS 6.
        initial=path(cfg.initial_adapter)
        tokenizer_path=str(initial) if adapter and (initial/'tokenizer_config.json').is_file() else model_path
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
        if cfg.chat_template_file:
            requested_template=path(cfg.chat_template_file).read_text()
            if trainable and self.tokenizer.chat_template and requested_template!=self.tokenizer.chat_template:
                raise ValueError('RL training chat template differs from the SFT checkpoint template')
            self.tokenizer.chat_template=requested_template
        self.tokenizer.pad_token=self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(model_path,
            dtype=torch.bfloat16 if cfg.precision=='bf16' else torch.float32,attn_implementation='sdpa')
        if cfg.context_limit>self.model.config.max_position_embeddings:
            raise ValueError('RL context limit exceeds model capability')
        if adapter:
            initial=path(cfg.initial_adapter)
            if not (initial/'adapter_config.json').is_file(): raise ValueError('SFT adapter not ready: '+str(initial))
            self.model=PeftModel.from_pretrained(self.model,str(initial),adapter_name='policy',is_trainable=trainable)
            if trainable:
                self.model.load_adapter(str(initial),adapter_name='reference',is_trainable=False)
                self.has_reference=True
                self.model.set_adapter('policy')
                self.freeze_reference()
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                self.model.enable_input_require_grads()
                # Long-context MLP intermediates otherwise exhaust small MIGs
                # during checkpoint recomputation. MLPs are token independent.
                from torch.utils.checkpoint import checkpoint
                mlp_chunk_size=1024 if cfg.device=='cuda' and torch.cuda.get_device_properties(0).total_memory>=40*2**30 else 128
                for module in self.model.modules():
                    if not all(hasattr(module,name) for name in ('gate_proj','up_proj','down_proj')):
                        continue
                    original_forward=module.forward
                    def chunked_mlp(x,_forward=original_forward,_chunk_size=mlp_chunk_size):
                        outputs=[]
                        for start in range(0,x.shape[-2],_chunk_size):
                            chunk=x[...,start:start+_chunk_size,:]
                            outputs.append(checkpoint(_forward,chunk,use_reentrant=False)
                                if torch.is_grad_enabled() else _forward(chunk))
                        return torch.cat(outputs,dim=-2)
                    module.forward=chunked_mlp

        elif trainable: raise ValueError('RL training must start from a retail SFT adapter')
        self.model.to(cfg.device)
        disable_dropout(self.model)
        self.model.eval()

    def freeze_reference(self):
        for name,p in self.model.named_parameters():
            if '.reference.' in name: p.requires_grad_(False)

    def sync_reference(self):
        """Freeze a round-start snapshot of the current policy in the ref adapter."""
        if not self.has_reference:
            raise RuntimeError('No frozen reference adapter')
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
        state=get_peft_model_state_dict(self.model,adapter_name='policy')
        set_peft_model_state_dict(self.model,state,adapter_name='reference')
        self.freeze_reference()

    @contextmanager
    def reference(self):
        if not self.has_reference: raise RuntimeError('No frozen reference adapter')
        self.model.set_adapter('reference')
        self.freeze_reference()
        try: yield
        finally:
            self.model.set_adapter('policy');self.freeze_reference()

    def generate(self,prepared,max_tokens,seed):
        import torch
        from transformers import GenerationConfig
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.cfg.device=='cuda': torch.cuda.manual_seed_all(seed)
        self.model.eval()
        ids=torch.tensor([prepared['prompt_ids']],device=next(self.model.parameters()).device)
        sampling=self.cfg.temperature>0
        generation=GenerationConfig(max_new_tokens=max_tokens,do_sample=sampling,
            **({'temperature':self.cfg.temperature,'top_k':0,'top_p':self.cfg.top_p} if sampling else {}),
            repetition_penalty=1.0,eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=True,output_scores=True,use_cache=True)
        with torch.inference_mode():
            result=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=generation,logits_to_keep=1)
            tokens=result.sequences[0,ids.shape[1]:]
            model_logs=[float(torch.log_softmax(score[0].float(),-1)[token]) for score,token in zip(result.scores,tokens,strict=True)]
            old=model_logs if sampling else [0.0]*len(tokens)
        return dict(completion_ids=tokens.tolist(),old_logprobs=old,model_token_logprobs=model_logs,
                    decoding_mode='sampling' if sampling else 'greedy',
                    top_p=self.cfg.top_p,top_k=0,repetition_penalty=1.0,
                    text=self.tokenizer.decode(tokens,skip_special_tokens=True),
                    ended_with_eos=bool(len(tokens) and tokens[-1].item()==self.tokenizer.eos_token_id))

    def score(self,action,reference=False,grad=False):
        import torch
        if reference:
            with self.reference(),torch.no_grad():
                self.model.eval()
                return action_logprobs(self.model,action['prompt_ids'],action['completion_ids'],self.cfg.temperature or 1.0,self.cfg.score_chunk_size)
        # Dropout is zero in both modes; train mode activates gradient checkpointing.
        self.model.train(grad)
        with torch.set_grad_enabled(grad):
            return action_logprobs(self.model,action['prompt_ids'],action['completion_ids'],self.cfg.temperature or 1.0,self.cfg.score_chunk_size)

    def score_batch(self,actions,reference=False,grad=False):
        import torch
        if len(actions)==1:return [self.score(actions[0],reference=reference,grad=grad)]
        def score():
            return action_logprobs_batch(self.model,actions,self.tokenizer.pad_token_id,
                                         self.cfg.temperature or 1.,self.cfg.score_chunk_size)
        if reference:
            with self.reference(),torch.no_grad():
                self.model.eval();return score()
        self.model.train(grad)
        with torch.set_grad_enabled(grad):return score()

    def generate_batch(self,prepared,max_tokens,seed):
        """One actual batched model.generate call; record the batch's sampling seed."""
        import torch
        from transformers import GenerationConfig
        if len(prepared)==1:return [self.generate(prepared[0],max_tokens,seed)]
        torch.manual_seed(seed)
        if self.cfg.device=='cuda':torch.cuda.manual_seed_all(seed)
        self.model.eval()
        device=next(self.model.parameters()).device
        width=max(len(p['prompt_ids']) for p in prepared)
        ids=torch.full((len(prepared),width),self.tokenizer.pad_token_id,device=device,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for row,p in enumerate(prepared):
            ids[row,-len(p['prompt_ids']):]=torch.tensor(p['prompt_ids'],device=device)
            mask[row,-len(p['prompt_ids']):]=1
        sampling=self.cfg.temperature>0
        config=GenerationConfig(max_new_tokens=max_tokens,do_sample=sampling,
            **({'temperature':self.cfg.temperature,'top_k':0,'top_p':self.cfg.top_p} if sampling else {}),
            repetition_penalty=1.,eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=True,output_scores=True,use_cache=True)
        with torch.inference_mode():
            output=self.model.generate(input_ids=ids,attention_mask=mask,generation_config=config,logits_to_keep=1)
            generated=output.sequences[:,width:]
            logs=[torch.log_softmax(scores.float(),-1).gather(-1,generated[:,step,None]).squeeze(-1).tolist()
                  for step,scores in enumerate(output.scores)]
        results=[]
        for row in range(len(prepared)):
            tokens=generated[row].tolist()
            if self.tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(self.tokenizer.eos_token_id)+1]
            probabilities=[values[row] for values in logs[:len(tokens)]]
            results.append(dict(completion_ids=tokens,old_logprobs=probabilities if sampling else [0.]*len(tokens),
                model_token_logprobs=probabilities,decoding_mode='sampling' if sampling else 'greedy',
                top_p=self.cfg.top_p,top_k=0,repetition_penalty=1.,
                text=self.tokenizer.decode(tokens,skip_special_tokens=True),
                ended_with_eos=bool(tokens and tokens[-1]==self.tokenizer.eos_token_id),
                batch_sampling_seed=seed,generation_batch_size=len(prepared)))
        return results
