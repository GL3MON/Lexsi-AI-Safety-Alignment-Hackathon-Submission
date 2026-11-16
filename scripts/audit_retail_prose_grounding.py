"""Check identifier copying and card last-four claims in supervised assistant prose."""
import argparse,json,re
from pathlib import Path

def collect_last_four(value):
 result=set()
 if isinstance(value,dict):
  for k,v in value.items():
   if k=='last_four':result.add(str(v))
   else:result.update(collect_last_four(v))
 elif isinstance(value,list):
  for v in value:result.update(collect_last_four(v))
 return result

def check(row):
 flags=[];visible='';last_four=set();selected=set(row['loss_message_indices']);faults={f['message_index'] for f in row.get('masked_faults',[])};failed_order_refs=set()
 for i,m in enumerate(row['messages']):
  if m['role'] in ('user','tool'):
   visible+='\n'+m['content']
   if m['role']=='tool':
    try:last_four.update(collect_last_four(json.loads(m['content'])))
    except ValueError:pass
  if i in faults:failed_order_refs.update(re.findall(r'#W\d+',m['content']))
  if m['role']!='assistant' or i not in selected:continue
  text=re.sub(r'<tool_call>.*?</tool_call>','',m['content'],flags=re.S)
  seen=set()
  checks=[('order_id',r'#W\d+'),('payment_id',r'\b(?:credit_card|gift_card|paypal|bank_transfer)_\d+\b'),('email',r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')]
  for kind,pattern in checks:
   for match in re.finditer(pattern,text):
    value=match.group()
    if kind=='order_id' and value in failed_order_refs and re.search(r'not found|failed|invalid|incorrect',text,re.I):continue
    if kind=='order_id' and re.fullmatch(r'#W0+',value) and re.search(r'look like|format|example',text[max(0,match.start()-70):match.start()],re.I):continue
    if value not in set(re.findall(pattern,visible)) and (kind,value) not in seen:flags.append({'id':row['id'],'message_index':i,'kind':kind,'value':value,'text':text,'known_identifiers':sorted(set(re.findall(pattern,visible)))});seen.add((kind,value))
  for match in re.finditer(r'\b(item(?:\s+IDs?|_ids?|\s+identifiers?)?|product(?:\s+IDs?|_ids?))\s*(?:is|are|:|#)?\s*(\d{6,})',text,re.I):
   kind='item_id' if match[1].lower().startswith('item') else 'product_id';value=match[2]
   if value not in set(re.findall(r'"'+kind+r'"\s*:\s*"(\d+)"',visible)) and (kind,value) not in seen:
    flags.append({'id':row['id'],'message_index':i,'kind':kind,'value':value,'text':text,'known_identifiers':sorted(set(re.findall(r'"'+kind+r'"\s*:\s*"(\d+)"',visible)))});seen.add((kind,value))
  for match in re.finditer(r'\b(?:ending(?:\s+in)?|last\s+(?:four|4)(?:\s+digits)?)\s*[:#]?\s*[*x•]*(\d+)',text,re.I):
   value=match[1]
   if (len(value)!=4 or value not in last_four) and ('card_last_four',value) not in seen:
    seen.add(('card_last_four',value));flags.append({'id':row['id'],'message_index':i,'kind':'card_last_four','value':value,'text':text,'known_last_four':sorted(last_four)})
 return flags

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/retail-data/retail-sdg-v2-700');p.add_argument('--output',default='outputs/diagnostics/retail-sdg-v2-final-audit/prose-grounding.json');a=p.parse_args();flags=[]
 for file in sorted((Path(a.source)/'records').glob('*.json')):flags.extend(check(json.loads(file.read_text())))
 from collections import Counter
 report={'flagged_conversations':len({f['id'] for f in flags}),'flagged_messages':len({(f['id'],f['message_index']) for f in flags}),'flag_counts':dict(Counter(f['kind'] for f in flags)),'flags':flags}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='flags'},indent=2))
 for f in flags[:12]:print(f['id'],f['message_index'],f['kind'],f['value'])
if __name__=='__main__':main()
