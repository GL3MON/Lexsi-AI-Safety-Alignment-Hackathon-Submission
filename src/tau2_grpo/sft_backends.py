"""AlignTune pre-masked SFT extension, with optional direct TRL diagnostics."""
import json
from pathlib import Path
from types import SimpleNamespace
from tau2_grpo.prepare_function_calling import digest
from tau2_grpo.aligntune_runtime import make_owner,verify_aligntune

class SFTBackend:
 def __init__(self,name,*,records,encoded,tokenizer,max_length,run_dir,config=None):
  if name=='agenttune':name='aligntune' # Old launch arguments remain readable; no old dependency.
  if name not in ('aligntune','trl'):raise ValueError('Unknown SFT backend: '+name)
  self.name,self.records,self.encoded=name,records,encoded
  self.tokenizer,self.max_length,self.run_dir=tokenizer,max_length,Path(run_dir)
  self.config=config or SimpleNamespace(model_id=getattr(tokenizer,'name_or_path','unknown'),dataset_path='verified-pretokenized-SFT')
  self.trainer=None;self.owner=None
  self.metadata={'backend':name,'train_fingerprint':digest(encoded),'precomputed_labels_preserved':True}
  if name=='aligntune':self.metadata.update(verify_aligntune())
 def train(self,*,build_trainer,on_ready,resume=None):
  try:
   self.trainer=build_trainer(self.encoded)
   if self.name=='aligntune':
    self.owner=make_owner(kind='masked_sft',config=self.config,run_dir=self.run_dir,model=getattr(self.trainer,'model',None),dataset=self.encoded,tokenizer=self.tokenizer,execute=lambda:self.trainer.train(resume_from_checkpoint=resume))
    if hasattr(self.trainer,'add_callback'):
     for callback in self.owner.get_hf_callbacks():self.trainer.add_callback(callback)
   on_ready(self.trainer)
   return self.owner.train() if self.owner else self.trainer.train(resume_from_checkpoint=resume)
  finally:
   self.run_dir.mkdir(parents=True,exist_ok=True)
   if self.owner:self.metadata.update(self.owner.metadata)
   (self.run_dir/'backend_audit.json').write_text(json.dumps(self.metadata,indent=2))
