"""Pinned AlignTune lifecycle for pre-masked SFT and turn-aware Tau2 GRPO.

Stock generation SFT drops our label columns; flat GRPO cannot replay causal
summary-conditioned turns. These explicit UnifiedTrainerBase extensions preserve
those contracts rather than pretending they are stock AlignTune algorithms.
"""
import json
import importlib.util
import subprocess
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse,unquote

ALIGNTUNE_REVISION='c0cb36e682d2fe212ad86a33da76f9fdd23ed9b6'

def verify_aligntune():
 package=distribution('aligntune');origin=json.loads(package.read_text('direct_url.json') or '{}')
 revision=origin.get('vcs_info',{}).get('commit_id')
 if origin.get('dir_info',{}).get('editable') and origin.get('url','').startswith('file:'):
  checkout=Path(unquote(urlparse(origin['url']).path))
  revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
  if subprocess.check_output(['git','-C',str(checkout),'diff','--name-only','HEAD'],text=True).strip():raise RuntimeError('Pinned AlignTune checkout has modified tracked files')
 if revision is None:
  # Editable setuptools metadata can resolve to source .egg-info, which has no
  # direct_url.json. Verify the actual imported checkout instead.
  spec=importlib.util.find_spec('aligntune')
  checkout=Path(spec.origin).resolve().parents[2] if spec and spec.origin else None
  if checkout and (checkout/'.git').exists():
   revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
   if subprocess.check_output(['git','-C',str(checkout),'diff','--name-only','HEAD'],text=True).strip():raise RuntimeError('Pinned AlignTune checkout has modified tracked files')
 if revision!=ALIGNTUNE_REVISION:raise RuntimeError(f'Expected AlignTune {ALIGNTUNE_REVISION}; installed origin is {revision}')
 return {'aligntune_version':package.version,'aligntune_revision':revision}

def make_owner(*,kind,config,run_dir,model,dataset,execute,tokenizer=None):
 """Build an actual AlignTune trainer with native callbacks and provenance."""
 metadata=verify_aligntune()
 from aligntune.core.trainer_base import UnifiedTrainerBase
 from aligntune.core.callbacks import CallbackHandler,TrainerCallback
 run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
 class Audit(TrainerCallback):
  def __init__(self):self.events=[]
  def record(self,event,**values):
   self.events.append({'event':event,**values});(run_dir/'aligntune_events.json').write_text(json.dumps(self.events,indent=2,default=str))
  def on_init_end(self,*args,**kwargs):self.record('init')
  def on_train_begin(self,*args,**kwargs):self.record('train_begin')
  def on_train_end(self,*args,**kwargs):self.record('train_end')
  def on_log(self,*args,logs=None,**kwargs):self.record('log',metrics=logs)
  def on_step_end(self,args,state,control,**kwargs):self.record('step_end',step=state.step)
 audit=Audit()
 # AlignTune's base accepts custom config objects; record original settings too.
 cfg=SimpleNamespace(model=SimpleNamespace(name_or_path=getattr(config,'model_id',None)),dataset=SimpleNamespace(name=getattr(config,'dataset_path',getattr(config,'split_manifest',None))),train=SimpleNamespace(learning_rate=getattr(config,'learning_rate',None)))
 class Tau2AlignTuneTrainer(UnifiedTrainerBase):
  def __init__(self):
   super().__init__(cfg,callbacks=[audit]);self.metadata={**metadata,'framework':'aligntune','extension':kind,'base_class':'aligntune.core.trainer_base.UnifiedTrainerBase'}
   self.setup_model();self.setup_data();self.tokenizer=tokenizer
   self.callback_handler=CallbackHandler(self.callbacks,self.model,self.tokenizer)
   self.control=self.callback_handler.on_init_end(self.config,self.state,self.control)
  def setup_model(self):self.model=model
  def setup_data(self):self.dataset=dataset
  def evaluate(self,*args,**kwargs):raise NotImplementedError('Evaluation is performed by the shared SFT/Tau2 evaluators')
  def _get_task_type(self):return kind
  def step(self,step,metrics):
   self.state.update_step(step);self.control=self.callback_handler.on_step_end(self.config,self.state,self.control)
   self.control=self.callback_handler.on_log(self.config,self.state,self.control,logs=metrics)
  def train(self):
   self.control=self.callback_handler.on_train_begin(self.config,self.state,self.control)
   try:
    result=execute();self.state.update_step(getattr(result,'global_step',self.state.step));self.write_provenance(run_dir)
    self.control=self.callback_handler.on_train_end(self.config,self.state,self.control)
    self.metadata['success']=True;return result
   except Exception:
    self.metadata['success']=False;raise
   finally:(run_dir/'aligntune_runtime.json').write_text(json.dumps(self.metadata,indent=2))
 return Tau2AlignTuneTrainer()
