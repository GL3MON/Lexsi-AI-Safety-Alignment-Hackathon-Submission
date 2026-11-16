import copy,unittest
import torch
from tau2_grpo.mlp_chunking import enable_mlp_chunking
class GatedMLP(torch.nn.Module):
 def __init__(self):
  super().__init__();self.gate_proj=torch.nn.Linear(7,13);self.up_proj=torch.nn.Linear(7,13);self.down_proj=torch.nn.Linear(13,7)
 def forward(self,x):return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))
class MLPChunkTests(unittest.TestCase):
 def test_outputs_input_gradients_and_weight_gradients_match(self):
  torch.manual_seed(13);plain=GatedMLP();chunked=copy.deepcopy(plain);self.assertEqual(enable_mlp_chunking(chunked,3),1)
  x=torch.randn(2,11,7,requires_grad=True);y=x.detach().clone().requires_grad_();a=plain(x);b=chunked(y)
  torch.testing.assert_close(a,b);a.square().sum().backward();b.square().sum().backward();torch.testing.assert_close(x.grad,y.grad)
  for pa,pb in zip(plain.parameters(),chunked.parameters()):torch.testing.assert_close(pa.grad,pb.grad)
  with torch.no_grad():torch.testing.assert_close(plain(x),chunked(x))
if __name__=='__main__':unittest.main()
