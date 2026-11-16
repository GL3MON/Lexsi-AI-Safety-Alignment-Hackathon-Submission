import json,copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch

class DPOLoopTests(unittest.TestCase):
 def test_two_updates_reference_cache_checkpoint_selection_and_final_adapter(self):
  from transformers import Cohere2Config,Cohere2ForCausalLM
  from peft import get_peft_model,LoraConfig
  from tau2_grpo.rl_policy import action_logprobs_batch
  from tau2_grpo.dpo import DPOSettings,train
  class Tokenizer:
   eos_token_id=2
   def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict):
    if add_generation_prompt:return [1,4,5]
    return [1,4,5,6 if messages[-1]['content']=='A' else 7,2]
   def save_pretrained(self,p):Path(p).mkdir(parents=True,exist_ok=True);(Path(p)/'tokenizer_config.json').write_text('{}')
  class Policy:
   def __init__(self,cfg,trainable):
    self.tokenizer=Tokenizer();self.model=get_peft_model(Cohere2ForCausalLM(Cohere2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,sliding_window=32,attention_dropout=0.,eos_token_id=2,bos_token_id=1,pad_token_id=0)),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'),adapter_name='policy')
    self.ref=copy.deepcopy(self.model);self.ref.requires_grad_(False)
   def score_batch(self,actions,reference=False,grad=False):
    model=self.ref if reference else self.model;model.train(grad)
    with torch.set_grad_enabled(grad):return action_logprobs_batch(model,actions,0)
  class Run:
   url='offline-test'
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def log(self,*a,**kw):pass
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);initial=d/'initial';initial.mkdir();(initial/'adapter_model.safetensors').write_bytes(b'fixture')
   for split,world in [('train','w1'),('eval','w2')]:
    records=[dict(id=split+str(i),world_id=world,category='fixture',prompt=[dict(role='user',content='test')],chosen='A',rejected='B',prompt_ids=[1,4,5],chosen_ids=[6,2],rejected_ids=[7,2]) for i in range(4)]
    (d/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
   cfg=DPOSettings(initial_adapter=str(initial),dataset_path=str(d/'train.jsonl'),eval_dataset_path=str(d/'eval.jsonl'),output_dir=str(d/'out'),device='cpu',precision='fp32',epochs=1,pairs_per_batch=1,gradient_accumulation_steps=2,eval_steps=1,wandb_mode='offline',learning_rate=.01)
   with patch('tau2_grpo.rl_policy.LocalPolicy',Policy),patch('wandb.init',return_value=Run()):train(cfg)
   result=json.loads((d/'out/result.json').read_text());self.assertTrue(result['complete']);self.assertEqual(result['optimizer_steps'],2)
   self.assertTrue((d/'out/final_adapter/adapter_model.safetensors').exists());self.assertTrue((d/'out/DPO_COMPLETE').exists())
   self.assertEqual(len(list((d/'out').glob('checkpoint-*'))),2)
if __name__=='__main__':unittest.main()
