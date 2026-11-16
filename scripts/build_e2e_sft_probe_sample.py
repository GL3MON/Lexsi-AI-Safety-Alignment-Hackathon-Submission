"""Cache the longest exact tokenized training example for the GPU memory probe."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.sft_data import tokenize_assistant_only

def main():
 root=Path('outputs/retail-data/retail-e2e-sft-retention');rows=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()];record=max(rows,key=lambda r:r.get('context_tokens',0));tok=AutoTokenizer.from_pretrained('outputs/rl-initialization/post20-6af9edf523985fb9cd839302');tok.chat_template=Path('templates/retail_sdg_v2.jinja').read_text();encoded,_=tokenize_assistant_only(tok,record['messages'],8192,record['loss_message_indices']);encoded['loss_weight']=1.;(root/'longest-tokenized.json').write_text(json.dumps(encoded));print(record['id'],len(encoded['input_ids']))
if __name__=='__main__':main()
