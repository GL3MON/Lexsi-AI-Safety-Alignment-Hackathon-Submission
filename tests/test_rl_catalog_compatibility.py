import copy,json,unittest
from tau2_grpo.rl_tau2 import validated_system_prompt

def tool(parameters):return {'type':'function','function':{'name':'lookup','description':'Source wording','parameters':parameters}}
class CatalogTests(unittest.TestCase):
 def setUp(self):self.schema={'type':'object','properties':{'email':{'type':'string'}},'required':['email']}
 def prompt(self,schema):return '<tools>'+json.dumps([tool(schema)])+'</tools>'
 def test_metadata_changes_preserve_exact_prompt(self):
  current=copy.deepcopy(self.schema);current['title']='Runtime';current['properties']['email'].update(title='Email',description='Current docs')
  prompt=self.prompt(self.schema);self.assertEqual(validated_system_prompt(prompt,[tool(current)]),prompt)
 def test_renamed_argument_is_rejected(self):
  with self.assertRaises(ValueError):validated_system_prompt(self.prompt(self.schema),[tool({'type':'object','properties':{'address':{'type':'string'}},'required':['address']})])
 def test_narrow_published_enum_is_supported(self):
  published=copy.deepcopy(self.schema);published['properties']['email']['enum']=['one@example.org']
  self.assertEqual(validated_system_prompt(self.prompt(published),[tool(self.schema)]),self.prompt(published))
 def test_runtime_narrowing_is_rejected(self):
  runtime=copy.deepcopy(self.schema);runtime['properties']['email']['enum']=['one@example.org']
  with self.assertRaises(ValueError):validated_system_prompt(self.prompt(self.schema),[tool(runtime)])
 def test_unadvertised_tool_is_rejected(self):
  extra=tool(self.schema);extra['function']['name']='unseen'
  with self.assertRaises(ValueError):validated_system_prompt(self.prompt(self.schema),[tool(self.schema),extra])
 def test_empty_required_is_equivalent_to_missing(self):
  self.assertEqual(validated_system_prompt(self.prompt({'type':'object','properties':{},'required':[]}),[tool({'type':'object','properties':{}})]),self.prompt({'type':'object','properties':{},'required':[]}))
if __name__=='__main__':unittest.main()
