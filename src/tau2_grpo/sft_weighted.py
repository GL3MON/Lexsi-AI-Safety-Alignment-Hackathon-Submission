"""Per-example mean NLL with explicit category weights and bounded head projection."""
def weighted_nll(hidden,labels,head,weights,chunk_size=32,logit_scale=1.,softcap=None):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    shifted=labels[:,1:];where=(shifted!=-100).nonzero(as_tuple=False)
    counts=(shifted!=-100).sum(-1)
    if (counts==0).any():raise ValueError('Weighted SFT sample has no target tokens')
    if not torch.isfinite(weights).all() or (weights<=0).any():raise ValueError('Invalid sample loss weight')
    batch_index,positions=where.unbind(-1)
    states=hidden[batch_index,positions];targets=shifted[batch_index,positions]
    scales=weights[batch_index].float()/counts[batch_index].float()/labels.shape[0]
    def project(h,t,scale):
        logits=F.linear(h.to(head.weight.dtype),head.weight,head.bias).float()*logit_scale
        if softcap is not None:logits=softcap*torch.tanh(logits/softcap)
        return (F.cross_entropy(logits,t,reduction='none')*scale).sum()
    pieces=[]
    for start in range(0,len(targets),chunk_size):
        args=(states[start:start+chunk_size],targets[start:start+chunk_size],scales[start:start+chunk_size])
        pieces.append(checkpoint(project,*args,use_reentrant=False) if torch.is_grad_enabled() else project(*args))
    return torch.stack(pieces).sum()

class WeightedCollator:
    def __init__(self,collator):self.collator=collator
    def __call__(self,features):
        import torch
        weights=[f['loss_weight'] for f in features]
        batch=self.collator([{k:v for k,v in f.items() if k!='loss_weight'} for f in features])
        batch['loss_weight']=torch.tensor(weights,dtype=torch.float32)
        return batch

def trainer_class():
    from trl import SFTTrainer
    class ExampleWeightedSFTTrainer(SFTTrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.data_collator=WeightedCollator(self.data_collator)
            # Our loss is a per-example batch mean, not a global token sum.
            # Transformers must retain its ordinary accumulation scaling.
            self.model_accepts_loss_kwargs=False
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            weights=inputs['loss_weight'];base=model.get_base_model() if hasattr(model,'get_base_model') else model
            kwargs={k:inputs[k] for k in ('input_ids','attention_mask','position_ids') if k in inputs}
            hidden=base.model(**kwargs,use_cache=False,return_dict=True).last_hidden_state
            loss=weighted_nll(hidden,inputs['labels'],base.lm_head,weights,
                chunk_size=getattr(self,'weighted_chunk_size',32),logit_scale=getattr(base,'logit_scale',1.),softcap=getattr(base.config,'final_logit_softcapping',None))
            return (loss,{'loss':loss}) if return_outputs else loss
    return ExampleWeightedSFTTrainer
