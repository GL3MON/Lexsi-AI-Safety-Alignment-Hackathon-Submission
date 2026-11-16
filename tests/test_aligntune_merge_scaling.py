import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/deps/aligntune/src'))

class AlignTuneScaleTests(unittest.TestCase):
 def test_merge_applies_scaled_delta_once(self):
  import torch
  from transformers import GPT2Config,GPT2LMHeadModel
  from peft import LoraConfig,get_peft_model
  from peft.tuners.lora.layer import LoraLayer
  from aligntune.core.merge import PEFTMerger
  for scale in (.25,.5,.75,1.):
   model=get_peft_model(GPT2LMHeadModel(GPT2Config(n_embd=8,n_layer=1,n_head=1,vocab_size=16)),LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn']))
   layer=next(x for x in model.modules() if isinstance(x,LoraLayer))
   original=layer.base_layer.weight.detach().clone()
   with torch.no_grad():
    layer.lora_A['default'].weight.fill_(.1);layer.lora_B['default'].weight.fill_(.2)
   layer.set_scale('default',scale)
   expected=original+layer.get_delta_weight('default')
   native=model.get_base_model()
   with tempfile.TemporaryDirectory() as d:PEFTMerger().merge_lora(model,d,legacy_config_fields=False)
   actual=native.transformer.h[0].attn.c_attn.weight
   torch.testing.assert_close(actual,expected)
   self.assertFalse(any(isinstance(x,LoraLayer) for x in native.modules()))

if __name__=='__main__':unittest.main()
