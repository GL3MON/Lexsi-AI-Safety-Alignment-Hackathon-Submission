import unittest
import torch
from torch.nn import functional as F
from tau2_grpo.sft_weighted import weighted_nll,WeightedCollator

class WeightedLossTests(unittest.TestCase):
 def test_chunked_loss_and_gradients_match_dense_per_example_objective(self):
  torch.manual_seed(17);h=torch.randn(3,9,5,requires_grad=True);head=torch.nn.Linear(5,13);head.requires_grad_(False)
  labels=torch.tensor([[-100,-100,1,2,-100,-100,-100,-100,-100],[-100,-100,3,4,5,6,7,8,9],[-100,10,9,8,-100,-100,-100,-100,-100]])
  w=torch.tensor([1.5,.5,1.]);loss=weighted_nll(h,labels,head,w,chunk_size=2,logit_scale=.75);g=torch.autograd.grad(loss,h)[0]
  dense_h=h.detach().clone().requires_grad_();parts=[]
  for i in range(3):
   y=labels[i,1:];mask=y!=-100;z=head(dense_h[i,:-1])*.75;parts.append(F.cross_entropy(z[mask],y[mask])*w[i])
  expected=torch.stack(parts).mean();expected_g=torch.autograd.grad(expected,dense_h)[0]
  torch.testing.assert_close(loss,expected);torch.testing.assert_close(g,expected_g)
 def test_longer_targets_do_not_get_more_weight(self):
  torch.manual_seed(3);head=torch.nn.Linear(4,9,bias=False);head.requires_grad_(False);state=torch.randn(4)
  h=state.repeat(2,8,1);labels=torch.tensor([[-100,2,-100,-100,-100,-100,-100,-100],[-100,2,2,2,2,2,2,2]])
  a=weighted_nll(h,labels,head,torch.ones(2),chunk_size=2)
  b=F.cross_entropy(head(state)[None],torch.tensor([2]));torch.testing.assert_close(a,b)
 def test_collator_preserves_weights_and_masks(self):
  def collate(features):
   self.assertTrue(all('loss_weight' not in f for f in features));return {'labels':torch.tensor([[-100,3],[-100,4]])}
  result=WeightedCollator(collate)([dict(input_ids=[1,3],labels=[-100,3],loss_weight=1.5),dict(input_ids=[1,4],labels=[-100,4],loss_weight=.5)])
  self.assertEqual(result['loss_weight'].tolist(),[1.5,.5]);self.assertEqual(result['labels'].tolist(),[[-100,3],[-100,4]])
 def test_no_target_sample_rejected(self):
  with self.assertRaises(ValueError):weighted_nll(torch.zeros(1,3,4),torch.full((1,3),-100),torch.nn.Linear(4,9),torch.ones(1))
if __name__=='__main__':unittest.main()

class WeightedTrainerIntegrationTests(unittest.TestCase):
 def test_real_peft_trainer_updates_and_evaluates(self):
  import tempfile,math
  from transformers import Cohere2Config,Cohere2ForCausalLM,PreTrainedTokenizerFast
  from tokenizers import Tokenizer
  from tokenizers.models import WordLevel
  from peft import get_peft_model,LoraConfig
  from datasets import Dataset
  from trl import SFTConfig
  from tau2_grpo.sft_weighted import trainer_class
  from tau2_grpo.sft_loss import configure_chunked_loss
  backend=Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,**{f'w{i}':i for i in range(4,64)}},unk_token='[UNK]'))
  tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
  model=Cohere2ForCausalLM(Cohere2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,sliding_window=32,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.))
  model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
  before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
  rows=[dict(input_ids=[1,4,5,6,7,2],labels=[-100,-100,-100,6,7,2],loss_weight=w) for w in [1.5,.5,1.,1.]]
  configure_chunked_loss(2,'native')
  with tempfile.TemporaryDirectory() as d:
   args=SFTConfig(output_dir=d,use_cpu=True,bf16=False,fp16=False,max_steps=2,per_device_train_batch_size=1,gradient_accumulation_steps=2,per_device_eval_batch_size=2,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},learning_rate=.01,loss_type='chunked_nll',dataset_kwargs={'skip_prepare_dataset':True},remove_unused_columns=False,report_to=[],save_strategy='no',logging_strategy='no',max_length=64)
   trainer=trainer_class()(model=model,args=args,processing_class=tok,train_dataset=Dataset.from_list(rows),eval_dataset=Dataset.from_list(rows));trainer.weighted_chunk_size=2
   result=trainer.train();metrics=trainer.evaluate()
   self.assertTrue(math.isfinite(result.training_loss));self.assertTrue(math.isfinite(metrics['eval_loss']))
   self.assertTrue(any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before))
