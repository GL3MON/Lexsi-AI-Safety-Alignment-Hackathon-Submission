import copy
import json
from pathlib import Path
import unittest

from scripts.prepare_apigen_retail_sft import normalize
from tau2_grpo.sft_data import encode_records


class APIGenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained('outputs/rl-glaive/run-55f242da3a/checkpoint-17/policy')
        cls.tokenizer.chat_template=Path('templates/retail_sdg_v2.jinja').read_text()
        cls.row=json.loads(next(Path('outputs/retail-data/apigen-mt-retail/retail.jsonl').open()))

    def test_real_error_and_correction_retained_but_failed_target_masked(self):
        row=copy.deepcopy(self.row)
        call=next(i for i,m in enumerate(row['conversations']) if m['from']=='function_call')
        bad=json.loads(row['conversations'][call]['value']);bad['name']='nonexistent_lookup'
        row['conversations'][call:call]=[{'from':'function_call','value':json.dumps(bad)},
            {'from':'observation','value':'Error: Unknown tool nonexistent_lookup'}]
        record,issue,encoded=normalize(row,0,self.tokenizer,8192)
        self.assertIsNone(issue)
        self.assertEqual(len(record['masked_faults']),1)
        bad_index=record['masked_faults'][0]['message_index']
        self.assertNotIn(bad_index,record['loss_message_indices'])
        self.assertIn(bad_index+2,record['loss_message_indices'])
        start=len(self.tokenizer.apply_chat_template(record['messages'][:bad_index],tokenize=True,add_generation_prompt=True))
        end=len(self.tokenizer.apply_chat_template(record['messages'][:bad_index+1],tokenize=True))
        self.assertTrue(all(label==-100 for label in encoded['labels'][start:end]))

    def test_unlisted_call_with_fabricated_success_is_not_recovery(self):
        row=copy.deepcopy(self.row)
        call=next(i for i,m in enumerate(row['conversations']) if m['from']=='function_call')
        bad=json.loads(row['conversations'][call]['value']);bad['name']='nonexistent_lookup'
        row['conversations'][call]['value']=json.dumps(bad)
        record,issue,_=normalize(row,0,self.tokenizer,8192)
        self.assertIsNone(record);self.assertEqual(issue['reasons'][0]['reason'],'unlisted_tool')

    def test_parallel_preprocessing_matches_serial_exact_labels(self):
        record,issue,_=normalize(self.row,0,self.tokenizer,8192)
        self.assertIsNone(issue)
        sequential=list(encode_records(self.tokenizer,[record],8192,1))
        parallel=list(encode_records(self.tokenizer,[record],8192,2))
        self.assertEqual(parallel,sequential)


if __name__=='__main__':unittest.main()
