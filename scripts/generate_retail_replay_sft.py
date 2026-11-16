"""Natural retail demonstrations with real-tool execution and masked corruption replay."""
import argparse, copy, json, random, re, time, traceback
from pathlib import Path
from openai import OpenAI, OpenAIError
from transformers import AutoTokenizer
from jsonschema import validate
from generate_long_horizon_recovery import make_world
from tau2_grpo.prepare_function_calling import parse_calls, digest
from tau2_grpo.sft_data import tokenize_assistant_only
from tau2_grpo.tool_messages import canonical_tool_messages
from tau2_grpo.dialogue_roles import assert_speaker
from tau2_grpo.arithmetic_validation import validate_calculation,calculation_satisfied,answer_matches,number
from tau2_grpo.rl_rollout import atomic_json

AGENT_SCHEMA={'type':'object','properties':{'speaker':{'type':'string','const':'agent'},'text':{'type':'string','minLength':1},'call':{'anyOf':[{'type':'null'},{'type':'object','properties':{'name':{'type':'string'},'arguments':{'type':'object','additionalProperties':True}},'required':['name','arguments'],'additionalProperties':False}]}},'required':['speaker','text','call'],'additionalProperties':False}
CUSTOMER_SCHEMA={'type':'object','properties':{'speaker':{'type':'string','const':'customer'},'text':{'type':'string','minLength':1}},'required':['speaker','text'],'additionalProperties':False}
KINDS=['json_structure','unknown_tool','argument_schema','wrong_identifier','wrong_operation']
READS={'get_user_details','get_order_details','get_product_details'}

def execute(tools,schemas,text):
    """Same parsing/schema checks as RetailEnvironment.step, then real RetailTools."""
    c=None;validated=False
    try:
        calls=parse_calls(text)
        if len(calls)!=1: raise ValueError('Execute exactly one call at a time and inspect its result')
        c=calls[0]
        if c['name'] not in schemas: raise ValueError('Unknown tool: '+c['name'])
        validate(c['arguments'],schemas[c['name']]);validated=True
        result=getattr(tools,c['name'])(**c['arguments'])
        body=result.model_dump(mode='json') if hasattr(result,'model_dump') else result
        if isinstance(body,str) and body.lower().startswith('error'): return c,body,False
        return c,body,True
    except Exception as e:
        return c if validated else None,{'error':str(e)[:600]},False

def corrupt(call,kind,rng,tools):
    c=copy.deepcopy(call)
    if kind=='json_structure':
        return '<tool_call>'+json.dumps(c)[:-rng.choice([1,2,3])]+'</tool_call>'
    if kind=='unknown_tool': c['name']=rng.choice(['lookup_customer_account','find_order_by_id','retrieve_order_information'])
    elif kind=='argument_schema':
        key=rng.choice(list(c['arguments']));value=c['arguments'].pop(key)
        c['arguments'][key+'_value']=value
    elif kind=='wrong_identifier':
        keys=[k for k,v in c['arguments'].items() if k.endswith('_id') and isinstance(v,str)]
        if not keys:return None
        key=rng.choice(keys);c['arguments'][key]='#W0000000' if key=='order_id' else 'nonexistent_0000000'
    elif kind=='wrong_operation':
        # A read-only intended order inspection is replaced with an impossible
        # cancellation of a delivered order, verified against a cloned database.
        if c['name']!='get_order_details':return None
        order=tools.db.orders.get(c['arguments'].get('order_id'))
        if order is None or order.status!='delivered':return None
        c={'name':'modify_pending_order_items','arguments':{'order_id':order.order_id,'item_ids':[order.items[0].item_id],'new_item_ids':[order.items[0].item_id],'payment_method_id':order.payment_history[0].payment_method_id}}
    return '<tool_call>'+json.dumps(c,separators=(',',':'))+'</tool_call>'

class Generator:
    def __init__(self,a):
        self.a=a;self.out=Path(a.output);self.out.mkdir(parents=True,exist_ok=True)
        for d in ['records','attempts','rejected','skipped']: (self.out/d).mkdir(exist_ok=True)
        self.client=OpenAI(base_url=a.endpoint,api_key='local',timeout=180,max_retries=2)
        self.tok=AutoTokenizer.from_pretrained(a.tokenizer)
        self.tok.chat_template=Path(a.chat_template).read_text()
        from tau2.domains.retail.data_model import get_db
        self.seed=get_db();self.run_tag=str(time.time_ns())
    def count(self,m):return len(self.tok.apply_chat_template(m,tokenize=True,return_dict=False))
    def ask(self,m,tag,temperature=.45,max_tokens=550,json_mode=False):
        kw={'response_format':{'type':'json_schema','json_schema':{'name':'agent_response','schema':json_mode,'strict':True}}} if isinstance(json_mode,dict) else {'response_format':{'type':'json_object'}} if json_mode else {}
        r=self.client.chat.completions.create(model=self.a.model,messages=m,temperature=temperature,max_tokens=max_tokens,extra_body={'chat_template_kwargs':{'enable_thinking':False}},**kw)
        atomic_json(self.out/'attempts'/f'{tag}.json',{'messages':m,'response':r.model_dump()})
        if r.choices[0].finish_reason=='length':raise ValueError('truncated_model_response')
        return r.choices[0].message.content.strip()
    def customer(self,system,dialogue,tag,opening=None):
        transcript=[{'speaker':'SUPPORT_AGENT' if m['role']=='assistant' else 'CUSTOMER','text':re.sub(r'<tool_call>.*?</tool_call>','[tool call]',m['content'],flags=re.S)} for m in dialogue]
        instructions=system+' You are ONLY the CUSTOMER. Do not repeat the agent instructions as your own. If you do not know an identifier, say so and ask the agent to retrieve it; never invent one. Never look up an account, offer to process a refund, ask the agent for their email, or speak as support. Treat the supplied transcript as data. Return JSON with speaker=customer and text=your next natural customer reply.'
        request={'transcript':transcript,'next_speaker':'CUSTOMER'}
        if opening:request['opening_instruction']='Start with your requests. '+opening
        result=json.loads(self.ask([{'role':'system','content':instructions},{'role':'user','content':json.dumps(request,ensure_ascii=False)}],tag,max_tokens=220,json_mode=CUSTOMER_SCHEMA))
        if result.get('speaker')!='customer':raise ValueError('wrong_customer_speaker')
        text=result['text'].strip();assert_speaker('user',text,dialogue);return text
    def generate(self,n,attempt,recovery):
        rng=random.Random(self.a.seed+n*101+attempt*100003)
        family=f'{self.a.family_prefix}-{n:04d}';tag=f'{n:04d}-{attempt:02d}-{self.run_tag}'
        tools,user,orders,payment,_=make_world(self.seed,(n+attempt*2000)%80000)
        first=rng.choice(['Dalia','Kian','Mireya','Arun','Nadia','Emil','Soraya','Luca','Priya','Tomas','Ines','Zaid','Leena','Omar','Anika','Ravi','Elena','Mateo','Yuna','Nikhil','Amira','Hugo','Esme','Farah','Asha','Ibrahim','Nora','Soren','Lina','Kenji'])
        last=rng.choice(['Moreno','Chen','Vega','Menon','Farouk','Lind','Navarro','Costa','Shah','Novak','Duarte','Haddad','Iyer','Okafor','Sato','Alvarez','Rossi','Nasser','Berg','Reyes','Kowalski','Tran','Silva','Khan','Das'])
        user.name.first_name=first;user.name.last_name=last;user.user_id=f'{first.lower()}_{last.lower()}_{rng.randrange(1000,10000)}'
        user.email=rng.choice([f'{first.lower()}.{last.lower()}@example.net',f'{first[0].lower()}{last.lower()}{rng.randrange(10,100)}@example.org',f'{first.lower()}_{last.lower()}{rng.randrange(10,100)}@example.com'])
        user.address.zip=f'{rng.randrange(10000,99999):05d}';tools.db.users={user.user_id:user}
        for order in orders:order.user_id=user.user_id;order.address=user.address.model_copy(deep=True)
        from tau2.domains.retail.utils import RETAIL_POLICY_PATH
        names=list(tools.get_tools())
        catalog=[t.openai_schema for t in tools.get_tools().values()]
        schemas={x['function']['name']:x['function']['parameters'] for x in catalog}
        system='Retrieve a verified account with get_user_details to find its orders; do not browse the product catalog to find customer orders. Include the tool call immediately when the next action is known. Ask only necessary questions. You are a helpful retail support agent. Follow the policy. Ask for missing account details; never invent arguments or identifiers. Every assistant turn MUST start with natural customer-facing text, including every tool-call turn. Never emit a bare tool block. Explain meaningful actions naturally and concisely. For tools emit exactly one <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>, with a brief customer-facing explanation when useful. Inspect actual results. After an error, the next response MUST begin with a short specific explanation of what failed and how you will fix it, followed by the corrected tool block. Do not merely resend JSON. For example: That request used an unsupported argument; I will resend it with the required field name. Use an explanation specific to the actual error; do not repeat this example. Then correct the request; do not repeat a failing call unchanged. Confirm modifications before execution. Only claim outcomes supported by successful tool results. Finish with a clear account of completed actions and any remaining requirements.\n\n'+RETAIL_POLICY_PATH.read_text()+'\n\n<tools>'+json.dumps(catalog,separators=(',',':'))+'</tools>'
        choices=[('cancel',orders[0]),('cancel',orders[1]),('return',orders[2]),('return',orders[3]),('address',orders[4]),('inspect',orders[5])]
        # Wrong-operation faults require a delivered-order inspection, not a fixed dialogue.
        kind=(self.a.corruption_kind or KINDS[n%5]) if recovery else None
        chosen=rng.sample(choices,rng.choice([1,1,2,2,3]))
        if kind=='wrong_operation' and not any(o.status=='delivered' for _,o in chosen):chosen.append(('inspect',orders[5]))
        calc_kind=self.a.calculator_scenario
        if calc_kind=='mixed':calc_kind=['refund_total','variant_difference','budget'][n%3]
        if calc_kind:chosen=[('inspect',orders[5])] if calc_kind=='variant_difference' else [('inspect',orders[2]),('inspect',orders[3])]
        # Keep only relevant orders so irrelevant database bulk does not consume context.
        tools.db.orders={o.order_id:o for _,o in chosen};user.orders=list(tools.db.orders)
        for o in tools.db.orders.values():
            o.items[0].price=round(35.5+(n*37+attempt*11+int(o.order_id[-1])*83)%1600,2)
            o.payment_history[0].amount=o.items[0].price
            tools.db.products[o.items[0].product_id].variants[o.items[0].item_id].price=o.items[0].price
        destination={'address1':f'{271+n} Juniper Street','address2':'','city':'Austin','state':'TX','country':'USA','zip':'78704'}
        goals=[];requests=[]
        for action,o in chosen:
            label=o.items[0].name
            if action=='cancel':request=f'Cancel my pending {label} order because I no longer need it.'
            elif action=='return':request=f'Return the delivered {label}, refunding the original payment method.'
            elif action=='address':request=f'Change the shipping address of my pending {label} order to '+json.dumps(destination)+'.'
            else:request=f'Tell me the status and price of my {label} order.'
            goals.append({'action':action,'order_id':o.order_id,'item_id':o.items[0].item_id,'price':o.items[0].price});requests.append(request)
        if calc_kind:
            if calc_kind=='variant_difference':
                order=chosen[0][1];product=tools.db.products[order.items[0].product_id];current=order.items[0].price
                alternative=next(v for k,v in product.variants.items() if k!=order.items[0].item_id and v.options.get('size')==order.items[0].options.get('size') and v.options.get('material')==order.items[0].options.get('material'))
                alternative.price=round(current+rng.choice([-17.25,23.75,41.5]),2)
                expected=round(abs(current-alternative.price),2)
                requests=['For my Hiking Boots, compare the price I paid with the other available variant in the same size and material. What is the price difference? Do not exchange or change anything yet.']
                goals.append({'action':'catalog','order_id':order.order_id,'product_id':product.product_id})
                calc_goal={'action':'calculate','kind':calc_kind,'expected':expected,'budget':None}
            else:
                total=round(sum(o.items[0].price for _,o in chosen),2)
                labels=' and '.join(o.items[0].name for _,o in chosen)
                if calc_kind=='refund_total':
                    requests=[f'Before I decide, what is the combined refund value if I return my {labels}? Please calculate the total. Do not submit any returns yet.']
                    calc_goal={'action':'calculate','kind':calc_kind,'expected':total,'budget':None}
                else:
                    budget=round(total+rng.choice([-75.5,-24.25,32.75,118.5]),2)
                    requests=[f'I have a budget of ${budget:.2f}. Would it cover the combined amount I paid for my {labels}? Calculate how much would be left or how far over budget that is. Do not change any orders.']
                    calc_goal={'action':'calculate','kind':calc_kind,'expected':round(budget-total,2),'budget':budget}
            goals.append(calc_goal)
            system+='\nFor arithmetic requested by the customer, retrieve the actual prices and use calculate. Calculator expressions must use the observed prices or the stated budget, never identifiers. Use calculate for the final requested numerical answer too: for a budget question, calculate the budget minus the combined prices, not just the combined prices. Do not compute the remaining balance mentally. Report the result and distinguish a hypothetical refund from a refund already paid.'
        facts={'first_name':user.name.first_name,'last_name':user.name.last_name,'zip':user.address.zip,'email':user.email,'orders':[{'order_id':o.order_id,'item':o.items[0].name} for _,o in chosen],'requests':requests}
        customer_system='Play only the retail customer. Use only these private facts: '+json.dumps(facts)+'. Pursue these requests. Reveal account details naturally when asked, without inventing values. Confirm the requested actions when the agent accurately describes them; correct misunderstandings. Do not write tool calls, tool results, policies, agent advice or generic placeholder identities. Use short natural replies. If finished, thank the agent.'
        opening=self.customer(customer_system,[],tag+'-opening',opening=rng.choice(['Do not provide account identity yet.','Include your email.','Include your name but wait to provide ZIP until asked.']))
        messages=[{'role':'system','content':system},{'role':'user','content':opening}];targets=[];audit=[];faults=[];auth=False;confirmed=False;pending_confirmation=False;last_error=False;corrected=False;dialogue_streak=0
        fault_target=rng.randint(1,1 if len(chosen)==1 and chosen[0][0]=='inspect' else 2);eligible=0
        fault_count=1
        fault_kind=kind
        for turn in range(30):
            if self.count(messages)>7600:raise ValueError('context_budget_before_completion')
            for draft_attempt in range(3):
                try:
                    teacher_messages=canonical_tool_messages(messages)
                    for ti,tm in enumerate(teacher_messages):
                        if tm['role']=='tool':tm.setdefault('tool_call_id',f'call-{ti-1}')
                    teacher_messages[0]['content']+='\n\nAUTHORING FORMAT: You are ONLY the SUPPORT AGENT. Never speak as the customer, copy their first-person request, or impersonate their identity. For this response return a JSON object with exactly three keys: speaker (always agent), text (a nonempty natural customer-facing explanation or answer) and call (null for a conversational answer/question, otherwise an object with name and arguments). The text must explain this specific action; after a failed call explain the specific error and repair. Do not put tool markup in text. The authoring system will render text and call into the normal conversation format. Do not use this authoring JSON inside text.'
                    if last_error:
                        teacher_messages[0]['content']+='\nRECOVERY REQUIREMENT: In text explicitly identify the cause of the last tool error (not just an apology or try-again statement), explain the specific change that fixes it, and emit the corrected call. The visible error and available schemas are your evidence.'
                    cg=next((g for g in goals if g['action']=='calculate'),None)
                    draft_schema=AGENT_SCHEMA
                    if cg and not calculation_satisfied(cg,audit) and self.goals_met([g for g in goals if g['action']!='calculate'],tools,destination,audit):
                        customer_numbers={number(x) for m in messages if m['role']=='user' and not m['content'].startswith('<tool_response>') for x in re.findall(r'(?<!\w)\d+(?:\.\d+)?',m['content'].replace(',',''))}
                        if cg.get('budget') is None or number(cg['budget']) in customer_numbers:
                            # Construction constraint only: the teacher still derives the
                            # expression from observed prices. No expected answer is supplied.
                            draft_schema=copy.deepcopy(AGENT_SCHEMA)
                            draft_schema['properties']['call']={'type':'object','properties':{'name':{'type':'string','const':'calculate'},'arguments':{'type':'object','properties':{'expression':{'type':'string'}},'required':['expression'],'additionalProperties':False}},'required':['name','arguments'],'additionalProperties':False}
                    if cg and not calculation_satisfied(cg,audit):
                        teacher_messages[0]['content']+='\nARITHMETIC REQUIREMENT: The final arithmetic answer requested by this customer has not yet been returned by calculate. Read the needed prices, then call calculate for that exact answer. For a budget question, after summing the costs, calculate budget minus that total. Do not answer the remaining balance mentally and do not ask whether the customer needs anything else yet.'
                    if draft_attempt:
                        hint='Your previous draft was rejected: '+draft_error+'. Use only visible facts and the listed schema.'
                        if 'include_needed_tool_call' in draft_error:hint='The requests are not complete. Do not merely announce a future tool action. Put the actual tool request in the call field now, or ask the specific missing question. A promise without a call does not execute anything.'
                        if 'confirmation' in draft_error:hint='Ask the customer for confirmation and set call=null. Execute only in a later turn after their confirmation.'
                        teacher_messages[0]['content']+='\nPRIVATE DRAFT REJECTION: '+hint
                    draft=json.loads(self.ask(teacher_messages,tag+f'-agent-{turn:02d}-draft-{draft_attempt}',max_tokens=512,json_mode=draft_schema))
                    if set(draft)!={'speaker','text','call'} or draft.get('speaker')!='agent' or not isinstance(draft['text'],str) or not draft['text'].strip():raise ValueError('invalid_agent_authoring_format')
                    text=draft['text'].strip()
                    assert_speaker('assistant',text,messages)
                    if '<tool_call>' in text:raise ValueError('markup_inside_authoring_text')
                    if draft['call'] is not None:text+='\n<tool_call>'+json.dumps(draft['call'],ensure_ascii=False,separators=(',',':'))+'</tool_call>'
                    from audit_retail_prose_grounding import check as check_prose
                    prose_issues=check_prose({'id':family,'messages':canonical_tool_messages(messages+[{'role':'assistant','content':text}]),'loss_message_indices':[len(messages)],'masked_faults':faults})
                    if prose_issues:raise ValueError('prose_identifiers_must_copy_observed_fields_exactly_and_payment_IDs_are_not_card_ending_digits')
                    calls=parse_calls(text)
                    if calls:
                        c0=calls[0]
                        if c0['name']=='calculate':
                            cg=next((g for g in goals if g['action']=='calculate'),None)
                            validate_calculation(c0['arguments'].get('expression',''),audit,cg.get('budget') if cg else None,'\n'.join(m['content'] for m in messages if m['role']=='user'))
                        if len(calls)!=1:raise ValueError('one_call_required')
                        if c0['name'].startswith(('cancel_','modify_','return_','exchange_')) and not confirmed:raise ValueError('ask_for_confirmation_in_separate_turn_before_write')
                        if not c0['name'].startswith('find_') and c0['name'] not in ('calculate',) and not auth:raise ValueError('ask_for_missing_identity_before_tools')
                        if c0['name'].startswith(('cancel_','modify_','return_','exchange_')):
                            goal=next((g for g in goals if g.get('order_id')==c0['arguments'].get('order_id')),None)
                            expected_name={'cancel':'cancel_pending_order','return':'return_delivered_order_items','address':'modify_pending_order_address'}.get(goal['action']) if goal else None
                            if c0['name']!=expected_name:raise ValueError('write_does_not_match_requested_action_for_this_order')
                            if goal['action']=='cancel':expected_args={'order_id':goal['order_id'],'reason':'no longer needed'}
                            elif goal['action']=='return':expected_args={'order_id':goal['order_id'],'item_ids':[goal['item_id']],'payment_method_id':payment}
                            else:expected_args={'order_id':goal['order_id'],**destination}
                            if c0['arguments']!=expected_args:raise ValueError('write_arguments_do_not_match_requested_items_payment_or_address')
                            if re.search(r'\b(?:have|has|already|successfully)\s+(?:been\s+)?(?:cancelled|canceled|returned|refunded|updated|modified|processed|completed|exchanged)\b',draft['text'],re.I):raise ValueError('do_not_claim_a_write_succeeded_before_its_tool_result')
                        if c0['name']=='transfer_to_human_agents':raise ValueError('these_supported_requests_do_not_need_escalation')
                        visible0='\n'.join(m['content'] for m in messages if m['role']=='user')
                        for key,value in c0['arguments'].items():
                            if key.endswith(('_id','_ids')) or c0['name'].startswith('find_'):
                                for v in value if isinstance(value,list) else [value]:
                                    if v!='original_payment' and str(v) not in visible0:raise ValueError('argument_not_grounded_in_visible_history:'+key)
                        _,_,valid0=execute(copy.deepcopy(tools),schemas,text)
                        if not valid0:raise ValueError('tool_request_is_invalid_check_schema_and_observed_state')
                        if last_error and c0!=faults[-1]['original_intended_call']:raise ValueError('repair_the_actual_failed_request_using_visible_history')
                    elif last_error:raise ValueError('provide_the_repaired_call_now')
                    elif cg and not calculation_satisfied(cg,audit) and all(g['action'] in ('inspect','catalog','calculate') for g in goals) and any(e['call']['name']=='calculate' for e in audit):raise ValueError('calculate_the_requested_final_answer_before_responding')
                    elif self.goals_met(goals,tools,destination,audit) and any(g['action']=='calculate' and not answer_matches(g,text) for g in goals):raise ValueError('final_answer_must_state_the_actual_calculator_amount_and_correct_budget_decision')
                    elif not self.goals_met(goals,tools,destination,audit) and not re.search(r'\?|please (?:provide|confirm|share)|could you|would you|may I',text,re.I):raise ValueError('include_needed_tool_call_or_ask_a_specific_question')
                    break
                except (ValueError,KeyError,TypeError) as e:
                    draft_error=str(e)
                    print('DRAFT RETRY',n,turn,draft_attempt,draft_error,flush=True)
                    if draft_attempt==2:raise
            if calls:
                if len(calls)!=1:raise ValueError('multiple_calls')
                dialogue_streak=0
                c=calls[0];is_write=c['name'].startswith(('cancel_','modify_','return_','exchange_'))
                if is_write and not confirmed:raise ValueError('write_without_confirmation')
                if not c['name'].startswith('find_') and c['name'] not in ('calculate','transfer_to_human_agents') and not auth:raise ValueError('tool_before_authentication')
                visible='\n'.join(m['content'] for m in messages if m['role']=='user')
                for k,v in c['arguments'].items():
                    if k.endswith(('_id','_ids')) or c['name'].startswith('find_'):
                        for value in v if isinstance(v,list) else [v]:
                            if value!='original_payment' and str(value) not in visible:raise ValueError('ungrounded_argument:'+k)
                # Verify intended call succeeds on an isolated snapshot before corrupting it.
                probe=copy.deepcopy(tools);_,_,valid=execute(probe,schemas,text)
                if not valid:raise ValueError('unplanned_agent_tool_failure')
                if recovery and len(faults)<fault_count and not last_error and c['arguments'] and not c['name'].startswith('find_'):
                    bad=corrupt(c,fault_kind,rng,tools)
                    if bad:
                        bad=re.sub(r'<tool_call>.*?</tool_call>',lambda _:bad,text,flags=re.S)
                        eligible+=1
                        if eligible>=fault_target or fault_kind=='wrong_operation':
                            trial=copy.deepcopy(tools);before=trial.db.model_dump(mode='json');failed_call,err,ok=execute(trial,schemas,bad)
                            if not ok and trial.db.model_dump(mode='json')==before:
                                idx=len(messages);messages.append({'role':'assistant','content':bad});messages.append({'role':'user','content':'<tool_response>'+json.dumps({'name':failed_call['name'] if failed_call else '__runtime__','content':err},separators=(',',':'))+'</tool_response>'})
                                faults.append({'message_index':idx,'kind':fault_kind,'original_intended_call':c,'database_unchanged':True,'loss_masked':True});last_error=True;eligible=0;fault_target=rng.randint(2,3);fault_kind=rng.choice(KINDS[:4])
                                continue
                call,result,ok=execute(tools,schemas,text)
                if not ok:raise ValueError('unexpected_tool_failure')
                if last_error:
                    patterns={'json_structure':r'JSON|syntax|brace|format|malformed|incomplete|structur','unknown_tool':r'unknown|unsupported|not available|isn.t available|unavailable|does not exist|doesn.t exist|unrecognized|incorrect|wrong|not a valid|not supported|not recognized','argument_schema':r'argument|parameter|field|schema|required','wrong_identifier':r'not found|identifier|incorrect|wrong|invalid|does not exist|couldn.t find','wrong_operation':r'delivered|pending|status|not allowed'}
                    prose=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S)
                    past_claim=r'\b(?:have|has|already|successfully)\s+(?:been\s+)?(?:cancelled|canceled|returned|refunded|updated|modified|processed|completed|exchanged)\b|corrected.{0,30}by.{0,20}(?:cancel|return|refund|updat|modif|process|exchang)'
                    if not re.search(patterns[faults[-1]['kind']],prose,re.I) or re.search(past_claim,prose,re.I):
                        # Teacher editorial pass improves narration only. The replay actor
                        # must already have independently produced the correct tool call.
                        narration=self.ask([{'role':'system','content':'Write only 1-3 natural customer-facing sentences preceding a repaired tool call. Explain the actual cause of the failed request and the specific repair. Do not merely apologize or say try again. The corrected tool call has NOT executed yet. Describe the repair as a FUTURE action using I will, I’ll, or Let me. Never say I have corrected this by cancelling, I have processed, or the action is complete. Do not invent facts, include tool markup, or change the supplied repair. This is a wording edit, not a request to choose a call.'},{'role':'user','content':json.dumps({'failed_assistant_response':messages[-2]['content'],'actual_error':messages[-1]['content'],'proposed_repair':call})}],tag+f'-recovery-narration-{turn:02d}',temperature=.5,max_tokens=200)
                        if not re.search(patterns[faults[-1]['kind']],narration,re.I) or not re.search(r'will|I.ll|let me|going to',narration,re.I) or re.search(past_claim,narration,re.I) or '<tool' in narration:raise ValueError('recovery_missing_specific_diagnosis')
                        text=narration+'\n<tool_call>'+json.dumps(call,ensure_ascii=False,separators=(',',':'))+'</tool_call>'
                    if call != faults[-1]['original_intended_call']:raise ValueError('recovery_did_not_restore_intended_call')
                    if not re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S).strip():raise ValueError('recovery_missing_explanation')
                    corrected=True;last_error=False
                if sum(x['call']==call for x in audit)>=2:raise ValueError('repeated_successful_call_loop')
                idx=len(messages);messages.append({'role':'assistant','content':text});targets.append(idx)
                messages.append({'role':'user','content':'<tool_response>'+json.dumps({'name':call['name'],'content':result},ensure_ascii=False,separators=(',',':'))+'</tool_response>'})
                if call['name'].startswith('find_'):auth=True
                # Confirmation may explicitly cover several actions; reviewer checks each write.
                audit.append({'message_index':idx,'call':call,'result':result})
            else:
                if last_error:raise ValueError('did_not_correct_on_next_turn')
                dialogue_streak+=1
                if dialogue_streak>3:raise ValueError('conversation_without_tool_progress')
                idx=len(messages);messages.append({'role':'assistant','content':text});targets.append(idx)
                done=self.goals_met(goals,tools,destination,audit)
                # Independent semantic audit requires supported final answer and confirmations.
                if done and any(g['action']=='calculate' and not answer_matches(g,text) for g in goals):raise ValueError('final_answer_must_state_the_verified_calculator_result_and_budget_decision')
                if done and (not recovery or (corrected and len(faults)==fault_count)):
                    verdict=json.loads(self.ask([{'role':'system','content':'Audit a retail demonstration. Return JSON {"accept":boolean,"reason":string}. Reject unsupported success/refund claims, absent explicit customer confirmation before changes, unmet requests, identifier inventions, repeated loops, speaker-role inversion (agent speaking as customer or customer acting as support), or an incomplete final answer. Tool returns are authoritative. For returns requested distinguish pending physical return from money already refunded. Do not reject a single deliberately injected failed call when the next assistant fixes it.'},{'role':'user','content':json.dumps({'goals':goals,'conversation':messages[1:]},ensure_ascii=False)}],tag+'-review',temperature=0,max_tokens=350,json_mode=True))
                    if verdict.get('accept') is not True:raise ValueError('semantic_review:'+str(verdict))
                    break
                pending_confirmation=bool(re.search(r'confirm|proceed|shall I|would you like|may I|permission|authorize',text,re.I))
                dialogue=[m for m in messages[1:] if not m['content'].startswith('<tool_response>')]
                response=self.customer(customer_system,dialogue,tag+f'-customer-{turn:02d}')
                messages.append({'role':'user','content':response})
                if pending_confirmation and re.search(r'yes|confirm|proceed|go ahead|please do|correct|sounds good',response,re.I):confirmed=True
        else:raise ValueError('turn_limit')
        tokens=self.count(messages)
        if tokens>8192:raise ValueError('over_8192')
        encoded,spans=tokenize_assistant_only(self.tok,messages,8192,targets)
        assistant_indices=[i for i,m in enumerate(messages) if m['role']=='assistant']
        for f in faults:
            start,end=spans[assistant_indices.index(f['message_index'])]
            assert all(v==-100 for v in encoded['labels'][start:end])
        assert len(encoded['input_ids'])==tokens
        assert all(f['message_index'] not in targets for f in faults)
        for fault in faults:
            i=fault['message_index']
            try: bad_calls=parse_calls(messages[i]['content'])
            except ValueError: bad_calls=[]
            assert bad_calls != [fault['original_intended_call']], 'original call paired with corrupted response'
            assert messages[i+1]['content'].startswith('<tool_response>')
            assert parse_calls(messages[i+2]['content']) == [fault['original_intended_call']]
            assert i+2 in targets and i not in targets
            result=json.loads(messages[i+3]['content'][15:-16])
            assert result['name']==fault['original_intended_call']['name']
            assert not (isinstance(result['content'],dict) and 'error' in result['content'])

        canonical=canonical_tool_messages(messages)
        canonical_encoded,_=tokenize_assistant_only(self.tok,canonical,8192,targets)
        assert canonical_encoded==encoded, 'Canonical tool roles changed token IDs or loss labels'
        messages=canonical
        split='validation' if n%50>=45 else 'train'
        return {'id':family,'task_id':family,'domain':'retail','group_id':digest({'domain':'retail','task_id':family}),'split':split,'messages':messages,'loss_message_indices':targets,'context_tokens':tokens,'masked_faults':faults,'tool_execution_audit':audit,'goals':goals,'semantic_review':verdict,'source_model':self.a.model,'mode':'recovery' if recovery else 'clean','seed':self.a.seed+n*101+attempt*100003,'chat_template_path':self.a.chat_template,'calculator_tool_choice_constrained_during_generation':bool(calc_kind)}
    @staticmethod
    def goals_met(goals,tools,destination,audit):
        for g in goals:
            if g['action']=='calculate':
                if not calculation_satisfied(g,audit):return False
                continue
            if g['action']=='catalog':
                if not any(e['call']['name']=='get_product_details' and e['call']['arguments'].get('product_id')==g['product_id'] for e in audit):return False
                continue
            o=tools.db.orders[g['order_id']];a=g['action']
            if a=='cancel' and o.status!='cancelled':return False
            if a=='return' and (o.status!='return requested' or g['item_id'] not in (o.return_items or [])):return False
            if a=='address' and any(getattr(o.address,k)!=v for k,v in destination.items()):return False
            if a=='inspect' and not any(x['call']['name']=='get_order_details' and x['call']['arguments'].get('order_id')==g['order_id'] for x in audit):return False
        return True
    def publish(self,stage,target):
        records=[json.loads(p.read_text()) for p in sorted((self.out/'records').glob('*.json'))]
        for name,split in [('sft','train'),('validation','validation'),('all',None)]:
            path=self.out/(name+'.jsonl');tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps({k:r[k] for k in ('id','task_id','domain','group_id','split','messages','loss_message_indices','context_tokens','mode','chat_template_path')},ensure_ascii=False)+'\n' for r in records if split is None or r['split']==split));tmp.replace(path)
        atomic_json(self.out/'task-split-manifest.json',{'partitions':{'sft':[r['task_id'] for r in records if r['split']=='train'],'validation':[r['task_id'] for r in records if r['split']=='validation']},'split_unit':'scenario_family'})
        report={'stage':stage,'completed':len(records),'recovery_completed':sum(r['mode']=='recovery' for r in records),'clean_completed':sum(r['mode']=='clean' for r in records),'target':target,'max_context_tokens':max((r['context_tokens'] for r in records),default=0),'updated_at':time.time()}
        atomic_json(self.out/'progress.json',report);print(json.dumps(report),flush=True)
    def run(self):
        saved=[json.loads(p.read_text()) for p in (self.out/'records').glob('*.json')]
        counts={mode:sum(r['mode']==mode for r in saved) for mode in ('recovery','clean')}
        n=self.a.start_index
        target=self.a.recovery_count+self.a.clean_count
        while counts['recovery']<self.a.recovery_count or counts['clean']<self.a.clean_count:
            path=self.out/'records'/f'{n:04d}.json'
            skipped=self.out/'skipped'/f'{n:04d}.json'
            if path.exists() or skipped.exists():n+=1;continue
            recovery=counts['recovery']<self.a.recovery_count;stage='recovery' if recovery else 'clean'
            for attempt in range(self.a.max_attempts):
                try:
                    r=self.generate(n,attempt,recovery)
                    if self.a.balanced_validation:r['split']='validation' if (counts[stage]+1)%10==0 else 'train'
                    from audit_retail_prose_grounding import check as check_prose
                    if check_prose(r):raise ValueError('final_prose_grounding_failed')
                    from audit_retail_dialogue_roles import audit as audit_roles
                    role_dir=self.out/'role-validation'/f'{n:04d}-{attempt:02d}-{self.run_tag}'
                    for sub in ['records','requests']:(role_dir/sub).mkdir(parents=True,exist_ok=True)
                    role_review=audit_roles(r,role_dir,self.a.endpoint)
                    if not role_review['role_consistent']:raise ValueError('final_role_consistency_failed:'+json.dumps(role_review['flags']))
                    r['role_review']=role_review
                    atomic_json(path,r)
                    path.with_suffix('.txt').write_text('\n\n'.join(('ASSISTANT [MASKED CORRUPTION]' if i in {f['message_index'] for f in r['masked_faults']} else 'TOOL RESPONSE' if m['role']=='tool' else m['role'].upper())+': '+m['content'] for i,m in enumerate(r['messages'])))
                    counts[stage]+=1
                    self.publish(stage,target);break
                except OpenAIError:
                    raise  # Infrastructure outages are not bad synthetic scenarios.
                except Exception as e:
                    atomic_json(self.out/'rejected'/f'{n:04d}-{attempt:02d}-{self.run_tag}.json',{'error':str(e),'traceback':traceback.format_exc()});print('REJECT',n,attempt,str(e),flush=True)
            else:
                atomic_json(skipped,{'scenario':n,'mode':stage,'attempts':self.a.max_attempts,'reason':'attempts_exhausted','skipped_at':time.time()})
                print('SKIP exhausted scenario',n,'continuing with a fresh scenario',flush=True)
                self.publish(stage,target)
            n+=1
        self.publish('complete',target)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/retail-data/retail-replay-2000');p.add_argument('--recovery-count',type=int,default=500);p.add_argument('--clean-count',type=int,default=1500);p.add_argument('--endpoint',default='http://127.0.0.1:8004/v1');p.add_argument('--model',default='qwen3.5-4b');p.add_argument('--tokenizer',default='outputs/hf-export/tiny-aya-global-function-calling-sft-525');p.add_argument('--chat-template',default='templates/retail_sdg_v2.jinja');p.add_argument('--start-index',type=int,default=0);p.add_argument('--seed',type=int,default=42631);p.add_argument('--max-attempts',type=int,default=8);p.add_argument('--calculator-scenario',choices=['refund_total','variant_difference','budget','mixed']);p.add_argument('--corruption-kind',choices=KINDS);p.add_argument('--family-prefix',default='retail-replay');p.add_argument('--balanced-validation',action='store_true')
    Generator(p.parse_args()).run()
if __name__=='__main__':main()
