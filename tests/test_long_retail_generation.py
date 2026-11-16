import unittest
from scripts.generate_long_retail_sft import prepare

class LongRetailStateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from tau2.domains.retail.data_model import get_db
  cls.base=get_db()

 def test_isolation_and_all_ten_action_profiles(self):
  from jsonschema import validate
  from tau2_grpo.prepare_function_calling import digest
  before=digest(self.base.model_dump(mode='json'))
  for n in range(10):
   tools,schemas,_,slots,steps,profile,action=prepare(self.base,n)
   schema={s['function']['name']:s['function']['parameters'] for s in schemas}
   errors=0;confirmed=False;authenticated=False
   for step in steps:
    if step[0]=='text':
     if step[2]=='approval':confirmed=True
     continue
    _,name,arguments=step;validate(arguments,schema[name])
    if name.startswith(('cancel_','return_')):self.assertTrue(confirmed)
    if not name.startswith('find_'):self.assertTrue(authenticated)
    try:
     getattr(tools,name)(**arguments)
     if name.startswith('find_'):authenticated=True
    except ValueError:
     errors+=1;self.assertEqual(profile,'failed_lookup');self.assertFalse(authenticated)
   self.assertEqual(errors,int(profile=='failed_lookup'))
   order=next(iter(tools.db.orders.values()))
   self.assertEqual(order.status,'cancelled' if action=='cancel' else 'return requested')
   if action=='cancel':
    payments=[p.amount for p in order.payment_history if p.transaction_type=='payment']
    refunds=[p.amount for p in order.payment_history if p.transaction_type=='refund']
    self.assertEqual(sum(payments),sum(refunds))
   else:self.assertCountEqual(order.return_items,[i.item_id for i in order.items])
  self.assertEqual(before,digest(self.base.model_dump(mode='json')))

 def test_bad_lookup_never_authenticates(self):
  tools,*_=prepare(self.base,0)
  with self.assertRaises(ValueError):tools.find_user_id_by_email('not-an-account@example.net')

if __name__=='__main__':unittest.main()
