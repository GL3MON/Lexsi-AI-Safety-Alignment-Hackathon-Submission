"""Build a strict no-failure subset of our generated retail SDG corpus."""
import argparse,json,re,shutil
import ast,itertools
from collections import Counter
from decimal import Decimal
from pathlib import Path

from audit_retail_prose_grounding import check as prose_flags


def derived_money(values,calls):
    result=set(values)
    for a,b in itertools.combinations(values,2):result.update((a+b,a-b,b-a))
    for a,b,c in itertools.combinations(values,3):
        result.update((a+b+c,a-b-c,b-a-c,c-a-b))
    for call in calls:
        if call['name']!='calculate':continue
        def evaluate(node):
            if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):
                number=Decimal(str(node.value))
                if number not in values:raise ValueError('unobserved arithmetic input')
                return number
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                value=evaluate(node.operand);return -value if isinstance(node.op,ast.USub) else value
            if isinstance(node,ast.BinOp):
                a,b=evaluate(node.left),evaluate(node.right)
                if isinstance(node.op,ast.Add):return a+b
                if isinstance(node.op,ast.Sub):return a-b
                if isinstance(node.op,ast.Mult):return a*b
                if isinstance(node.op,ast.Div):return a/b
            raise ValueError('unsupported arithmetic')
        try:result.add(evaluate(ast.parse(call['arguments']['expression'],mode='eval').body))
        except (ValueError,KeyError,SyntaxError,ArithmeticError):pass
    return result


def money(value):
    found=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if key in ('price','amount','balance','price_difference') and isinstance(child,(int,float)):
                found.add(Decimal(str(child)))
            found.update(money(child))
    elif isinstance(value,list):
        for child in value:found.update(money(child))
    elif isinstance(value,str):
        found.update(Decimal(s.replace(',','')) for s in re.findall(r'\$([0-9][0-9,]*(?:\.[0-9]+)?)',value))
        if re.fullmatch(r'-?\d+(?:\.\d+)?',value.strip()):found.add(Decimal(value.strip()))
    return found


def strict_flags(row,allow_recovery=False):
    from tau2_grpo.prepare_function_calling import parse_calls
    from jsonschema import Draft7Validator
    flags=[]
    if not allow_recovery and (row.get('mode')!='clean' or row.get('masked_faults')):
        return [dict(kind='intentional_recovery_context',detail='Kept separately; strict clean set contains no failed calls')]
    flags.extend(prose_flags(row))
    faults={f['message_index']:f for f in row.get('masked_faults',[])}
    catalog=json.loads(row['messages'][0]['content'].split('<tools>')[1].split('</tools>')[0])
    validators={t['function']['name']:Draft7Validator(t['function']['parameters']) for t in catalog}
    targets=set(row['loss_message_indices'])
    if targets!={i for i,m in enumerate(row['messages']) if m['role']=='assistant'}-set(faults):flags.append(dict(kind='unexpected_loss_mask'))
    for index,fault in faults.items():
        if not fault.get('loss_masked') or not fault.get('database_unchanged') or index in targets:
            flags.append(dict(kind='invalid_recovery_mask_or_state',message_index=index))
        try:
            failed=parse_calls(row['messages'][index]['content'])
        except ValueError:failed=[]
        try:
            repaired=parse_calls(row['messages'][index+2]['content'])
            if row['messages'][index+1]['role']!='tool' or index+2 not in targets or repaired!=[fault['original_intended_call']] or failed==repaired:
                flags.append(dict(kind='invalid_recovery_pair',message_index=index))
        except (IndexError,ValueError):flags.append(dict(kind='invalid_recovery_pair',message_index=index))
    observed_money=set();authenticated=False;latest_order=None;pending_confirmation=set();confirmed=set();completed=set();order_items={}
    for i,message in enumerate(row['messages']):
        text=message['content']
        if message['role']=='tool':
            result=json.loads(text);content=result['content'];name=result['name']
            failed=(isinstance(content,dict) and bool(content.get('error'))) or (isinstance(content,str) and bool(re.search(r'^error\b|not found|unknown tool|invalid argument',content,re.I)))
            if failed:
                if i-1 not in faults:flags.append(dict(kind='unmasked_failed_tool_response',message_index=i))
                continue
            observed_money.update(money(content))
            if name=='calculate' and isinstance(content,(int,float)):observed_money.add(Decimal(str(content)))
            if name.startswith('find_user_id_') and not failed:authenticated=True
            if name=='get_order_details' and isinstance(content,dict):
                latest_order=content.get('order_id');order_items[latest_order]={item['name'].lower() for item in content.get('items',[])}
            if name.startswith(('cancel_','modify_','return_','exchange_')) and not failed:completed.add(name)
        elif message['role']=='user':
            observed_money.update(money(text))
            for amount in re.findall(r'\bbudget\b.{0,35}?\$?([0-9][0-9,]*(?:\.\d{1,2})?)',text,re.I):
                observed_money.add(Decimal(amount.replace(',','')))
            negative=re.search(r"^\s*no\b|\b(?:do not|don't|not)\s+(?:cancel|return|modify|change|exchange|proceed|confirm)\b|\b(?:instead|except)\b",text,re.I)
            if pending_confirmation and re.search(r'\b(?:yes|confirm|proceed|go ahead|please do|correct|sounds good)\b',text,re.I) and not negative:
                confirmed.update(pending_confirmation)
            pending_confirmation=set()
        elif message['role']=='assistant':
            if i in faults:continue
            try:calls=parse_calls(text)
            except ValueError:
                flags.append(dict(kind='invalid_tool_json',message_index=i));continue
            for call in calls:
                name=call['name']
                if name not in validators or not validators[name].is_valid(call['arguments']):flags.append(dict(kind='invalid_tool_schema',message_index=i))
                if name.startswith(('cancel_','modify_','return_','exchange_')):
                    order=call['arguments'].get('order_id')
                    if order is not None and order not in confirmed:flags.append(dict(kind='missing_scoped_confirmation',message_index=i,order_id=order))
                    confirmed.discard(order)
            prose=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S)
            for amount in money(prose)-derived_money(observed_money,calls):flags.append(dict(kind='unsupported_money_in_prose',message_index=i,value=str(amount)))
            if not authenticated and re.search(r"(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:found|verified|located|authenticated)\s+(?:your|the)\s+(?:account|identity|user)",prose,re.I):
                flags.append(dict(kind='premature_authentication_claim',message_index=i))
            for verb,tool_prefix in [('cancel','cancel_'),('return','return_'),('update','modify_'),('modif','modify_'),('exchange','exchange_')]:
                positive_sentences=[sentence for sentence in re.split(r'(?<=[.!?])\s+',prose) if not re.search(r'\b(?:no|neither|not)\b',sentence,re.I)]
                has_execution=any(t.startswith(tool_prefix) for t in completed)
                if verb=='update' and 'status' in prose.lower() and completed:has_execution=True
                if any(re.search(r'\b(?:have|has|already|successfully)\s+(?:been\s+)?'+verb,s,re.I) for s in positive_sentences) and not has_execution:
                    flags.append(dict(kind='premature_success_claim',message_index=i,action=verb))
            if not calls and re.search(r'confirm|proceed|shall I|would you like|may I|permission|authorize',prose,re.I):
                mentions={oid for oid,names in order_items.items() if any(name in prose.lower() for name in names)}
                pending_confirmation=set(re.findall(r'#W\d+',prose)) or mentions or ({latest_order} if latest_order else set())
    return flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/retail-data/retail-sdg-v2-700'))
    parser.add_argument('--output',type=Path,default=Path('outputs/retail-data/retail-sdg-v2-strict-clean'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'records').mkdir(exist_ok=True);retained=[];excluded=[];recovery=[]
    for file in sorted((args.source/'records').glob('*.json')):
        row=json.loads(file.read_text());flags=strict_flags(row)
        if flags:
            excluded.append(dict(id=row['id'],mode=row['mode'],split=row['split'],reasons=flags,source=str(file)))
            if row['mode']=='recovery' and not prose_flags(row):recovery.append(row)
        else:
            retained.append(row);shutil.copyfile(file,args.output/'records'/file.name)
    for name,rows in [('all',retained),('sft',[r for r in retained if r['split']=='train']),
                      ('validation',[r for r in retained if r['split']=='validation']),('recovery-separate',recovery)]:
        (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (args.output/'excluded-manifest.json').write_text(json.dumps(excluded,indent=2))
    manifest=dict(source=str(args.source),source_records=len(retained)+len(excluded),retained=len(retained),
        train=sum(r['split']=='train' for r in retained),validation=sum(r['split']=='validation' for r in retained),
        excluded=len(excluded),separate_recovery_candidates=len(recovery),
        reason_counts=dict(Counter(f['kind'] for r in excluded for f in r['reasons'])),
        original_messages_and_targets_unchanged=True,original_splits_preserved=True,
        no_failed_tool_calls=True,scope='Automated schema, error-response, prose-grounding and confirmation checks; real replay and fresh label verification run separately.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
