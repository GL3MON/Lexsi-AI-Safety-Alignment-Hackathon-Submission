from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy


class PromptContractTests(unittest.TestCase):
    def test_mismatched_training_template_rejected_before_weights_load(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'tokenizer_config.json').write_text('{}')
            wrong=root/'wrong.jinja';wrong.write_text('wrong template')
            cfg=replace(RetailRLConfig(),initial_adapter=temp,model_id='unused',chat_template_file=str(wrong))
            tokenizer=SimpleNamespace(chat_template='trained template')
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained') as load:
                with self.assertRaisesRegex(ValueError,'differs from the SFT'):
                    LocalPolicy(cfg,trainable=True)
                load.assert_not_called()

    def test_checkpoint_tokenizer_used_without_an_explicit_template_override(self):
        with TemporaryDirectory() as temp:
            (Path(temp)/'tokenizer_config.json').write_text('{}')
            cfg=replace(RetailRLConfig(),initial_adapter=temp,model_id='unused',chat_template_file='')
            tokenizer=SimpleNamespace(chat_template='trained template',pad_token='pad',eos_token='eos')
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer) as load_tokenizer, \
                 patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=RuntimeError('weights boundary')):
                with self.assertRaisesRegex(RuntimeError,'weights boundary'):
                    LocalPolicy(cfg,trainable=True)
                load_tokenizer.assert_called_once_with(temp)


if __name__=='__main__':unittest.main()
