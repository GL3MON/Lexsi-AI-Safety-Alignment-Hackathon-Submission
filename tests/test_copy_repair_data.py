import unittest
from scripts.generate_copy_repair_sft import email_for,scenario

class CopyRepairTests(unittest.TestCase):
 def test_diverse_unique_email_formats(self):
  values=[email_for(n,'Lena','Avery') for n in range(500)]
  self.assertEqual(len(values),len(set(values)))
  domains={s.split('@')[-1] for s in values}
  self.assertGreaterEqual(len(domains),6)
  self.assertTrue(any('+' in s for s in values))
  self.assertTrue(any('_' in s for s in values))
  self.assertTrue(any('-' in s for s in values))
  self.assertTrue(any(not any(c.isdigit() for c in s.split('@')[0]) for s in values))
 def test_tool_execution_for_all_profiles(self):
  from tau2.domains.retail.data_model import get_db
  base=get_db()
  for n in range(10):
   tools,_,_,slots,steps,profile=scenario(base,n)
   errors=0
   for step in steps:
    if step[0]!='call':continue
    _,name,args,_,last=step
    try:getattr(tools,name)(**args)
    except ValueError:
     errors+=1;self.assertEqual(profile,'corrected_email');self.assertFalse(last)
   self.assertEqual(errors,int(profile=='corrected_email'))
 def test_zip_only_questions_never_request_known_name(self):
  from tau2.domains.retail.data_model import get_db
  for n in (6,7):
   _,_,_,slots,_,_=scenario(get_db(),n)
   self.assertEqual(slots['ask_zip']['text'],'What is your ZIP code?')

if __name__=='__main__':unittest.main()
