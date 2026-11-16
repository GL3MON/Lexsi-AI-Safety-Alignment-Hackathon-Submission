import importlib.util,json
from pathlib import Path
import sys,unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from clean_generated_retail_sft import strict_flags


class CleanRetailTests(unittest.TestCase):
    def kinds(self,identifier):
        row=json.loads((Path('outputs/retail-data/retail-sdg-v2-700/records')/(identifier.rsplit('-',1)[-1]+'.json')).read_text())
        return {f['kind'] for f in strict_flags(row)}

    def test_no_longer_needed_does_not_invalidate_explicit_confirmation(self):
        self.assertNotIn('missing_scoped_confirmation',self.kinds('retail-sdg-v2-100017'))

    def test_observed_return_status_update_is_not_an_unsupported_address_write(self):
        self.assertNotIn('premature_success_claim',self.kinds('retail-sdg-v2-130042'))

    def test_supported_budget_arithmetic_is_allowed(self):
        self.assertNotIn('unsupported_money_in_prose',self.kinds('retail-sdg-v2-140000'))

    def test_price_decimal_copy_error_is_rejected(self):
        self.assertIn('unsupported_money_in_prose',self.kinds('retail-sdg-v2-120037'))

    def test_asking_confirmation_and_executing_without_waiting_is_rejected(self):
        self.assertIn('missing_scoped_confirmation',self.kinds('retail-sdg-v2-100030'))


if __name__=='__main__':unittest.main()
