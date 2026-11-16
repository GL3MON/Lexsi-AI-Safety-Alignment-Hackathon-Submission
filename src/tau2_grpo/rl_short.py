"""Short verifiable read-only subgoals derived from reserved retail tasks.

These rewards are curriculum checks, not native Tau2 task-success scores.
"""
import copy,json,re
from pathlib import Path
from tau2_grpo.prepare_function_calling import parse_calls
from tau2_grpo.rl_config import path

KINDS=('email_lookup','name_zip_lookup','missing_zip','missing_last_name','missing_first_name','failed_email','get_profile','order_status')
FIELDS={'zip':r'\bzip\b|postal','first_name':r'first name|given name','last_name':r'last name|surname|family name','email':r'e-?mail'}

def clarification_ok(text,field):
    if '<tool_call>' in text:return False
    lower=text.lower()
    if re.search(r'i (?:have |already )?(?:found|located|verified|authenticated)|here are|status is|shipping address|\buser\s*:|\bassistant\s*:',lower):return False
    request=' '.join(re.findall(r'(?:could you|can you|would you|please|what (?:is|are)|which|may i|do you)[^?.!]*(?:[?.!]|$)',lower))
    if not re.search(FIELDS[field],request):return False
    if any(re.search(pattern,request) for other,pattern in FIELDS.items() if other!=field):return False
    if re.search(r'other|additional|relevant|full name|anything else',request):return False
    if field=='email' and not re.search(r'check|correct|confirm|verify|typo|spelling',request):return False
    return True


class ShortRetailEnvironment:
    def __init__(self,task_id,seed,cfg,tools=None,order_id=None):
        self.task_id=str(task_id);self.cfg=cfg;self.seed=seed;self.tools=tools;self.order_id=order_id;self.checks=[];self.done=False;self.position=0;self.success=False
    def reset(self):
        from tau2.domains.retail.data_model import get_db
        from tau2.domains.retail.tools import RetailTools
        if self.tools is None:
            base=get_db().model_copy(deep=True)
            from tau2.domains.retail.utils import RETAIL_TASK_SET_PATH
            tasks=json.loads(RETAIL_TASK_SET_PATH.read_text())
            task=next(t for t in tasks if str(t['id'])==self.task_id)
            self.order_id=next(a['arguments']['order_id'] for a in task['evaluation_criteria']['actions'] if a.get('arguments',{}).get('order_id') in base.orders)
            self.tools=RetailTools(base)
        order=self.tools.db.orders[self.order_id];customer=self.tools.db.users[order.user_id]
        self.customer=customer;self.order=order;kind=self.cfg.short_scenario
        if kind not in KINDS:raise ValueError('Unknown short scenario: '+kind)
        system=path(self.cfg.system_prompt_file).read_text();catalog=json.loads(re.findall(r'<tools>(.*?)</tools>',system,re.S)[0]);self.schemas={t['function']['name']:t['function']['parameters'] for t in catalog}
        messages=[dict(role='system',content=system)]
        def call(name,args):return dict(name=name,arguments=args)
        self.email_call=call('find_user_id_by_email',dict(email=customer.email))
        self.name_call=call('find_user_id_by_name_zip',dict(first_name=customer.name.first_name,last_name=customer.name.last_name,zip=customer.address.zip))
        def user(text):messages.append(dict(role='user',content=text))
        def observed(c,error=False):
            try:
                value=getattr(self.tools,c['name'])(**c['arguments']);body=value.model_dump(mode='json') if hasattr(value,'model_dump') else value
                if error:raise RuntimeError('Expected failed lookup')
            except ValueError as exc:
                if not error:raise
                body='Error: '+str(exc)
            messages.append(dict(role='assistant',content='<tool_call>'+json.dumps(c,separators=(',',':'))+'</tool_call>'))
            messages.append(dict(role='tool',name=c['name'],content=json.dumps(dict(name=c['name'],content=body),separators=(',',':'))))
        first,last,zip_code=customer.name.first_name,customer.name.last_name,customer.address.zip
        self.reply=None
        if kind=='email_lookup':user(f'Please find my account using this email: {customer.email}.');self.expected=[self.email_call]
        elif kind=='name_zip_lookup':user(f'I cannot remember my email. Please find my account. My first name is {first}, last name is {last}, and ZIP is {zip_code}.');self.expected=[self.name_call]
        elif kind.startswith('missing_'):
            field=kind[len('missing_'):]
            known={'zip':f'My first name is {first} and last name is {last}.','last_name':f'My first name is {first} and ZIP is {zip_code}.','first_name':f'My last name is {last} and ZIP is {zip_code}.'}[field]
            user('Please find my account. I cannot remember my email. '+known)
            label={'zip':'ZIP code','last_name':'last name','first_name':'first name'}[field];value={'zip':zip_code,'first_name':first,'last_name':last}[field]
            self.reply=f'My {label} is {value}.';self.expected=['ask:'+field,self.name_call]
        elif kind=='failed_email':
            bad=customer.email.replace('@','-typo@');user(f'Please find my account using {bad}.');observed(call('find_user_id_by_email',dict(email=bad)),True)
            self.reply=f'The correct email is {customer.email}. Please use this corrected email.';self.expected=['ask:email',self.email_call]
        elif kind=='get_profile':
            user(f'Please retrieve my account details. My email is {customer.email}.');observed(self.email_call);self.expected=[call('get_user_details',dict(user_id=customer.user_id))]
        else:
            # Specify a real order so retrieval is unambiguous, while requiring
            # authentication and the returned order list before accessing it.
            user(f'Please authenticate my account and check the current status of order {self.order_id}. My email is {customer.email}. Do not change anything.')
            self.expected=[self.email_call,call('get_user_details',dict(user_id=customer.user_id)),call('get_order_details',dict(order_id=self.order_id)),'report_status']
        self.history=copy.deepcopy(messages)
        return messages
    def step(self,text):
        expected=self.expected[self.position];parsed=[];error=None;incoming=[]
        try:parsed=parse_calls(text)
        except ValueError as exc:error=str(exc)
        if isinstance(expected,str) and expected.startswith('ask:'):
            ok=error is None and clarification_ok(text,expected.split(':',1)[1])
            visible='\n'.join(m['content'] for m in self.history if m['role'] in ('user','tool'))
            if any(v not in visible for v in re.findall(r'[\w.+-]+@[\w.-]+|#W\d+|\b\d[\d.,]*\b',text)):ok=False
            if re.search(r'\bcancel|\brefund|\bexchange|\bbalance|\bshipped',text,re.I):ok=False
            if ok:incoming=[dict(role='user',content=self.reply)]
        elif expected=='report_status':
            ok=error is None and not parsed and self.order_id in text and str(self.order.status).lower() in text.lower() and not re.search(r'\buser\s*:|\bassistant\s*:',text,re.I)
        else:
            ok=error is None and parsed==[expected]
            if ok:
                try:
                    value=getattr(self.tools,expected['name'])(**expected['arguments']);body=value.model_dump(mode='json') if hasattr(value,'model_dump') else value
                    incoming=[dict(role='tool',name=expected['name'],content=json.dumps(dict(name=expected['name'],content=body),separators=(',',':')))]
                except ValueError as exc:ok=False;error=str(exc)
        if re.search(r'\buser\s*:|\bassistant\s*:|\bcustomer\s*:',text,re.I):ok=False
        self.checks.append(dict(turn=self.position,criterion='clarification' if isinstance(expected,str) and expected.startswith('ask:') else 'status_report' if expected=='report_status' else 'exact_grounded_call',expected=expected,passed=bool(ok),error=error))
        self.position+=1
        self.done=not ok or self.position==len(self.expected);self.success=self.done and bool(ok) and self.position==len(self.expected)
        if not ok:incoming=[dict(role='tool',name='__runtime__',error=True,content=json.dumps(dict(name='__runtime__',content={'error':'Incorrect short-task next action; use supplied information and the declared tools.'})))]
        self.history.append(dict(role='assistant',content=text));self.history.extend(copy.deepcopy(incoming))
        return incoming,self.done,parsed[0] if parsed and len(parsed)==1 else None
    def finish(self):
        # No partial credit for filler or for stopping before a required call.
        # A correct short task beats every unsuccessful trajectory.
        reward=1.0 if self.success else -1.0
        self.reward_details=dict(version='short-retail-verifiable-v1',scenario=self.cfg.short_scenario,short_success=self.success,checks=self.checks,total=reward,note='Reserved-task-derived short curriculum reward; not a native Tau2 task-success score.')
        simulation=dict(reward_info=dict(reward_basis=['SHORT_SUBGOAL'],short_checks=self.checks),short_success=self.success,source_task_id=self.task_id,scenario=self.cfg.short_scenario)
        return reward,simulation
    def close(self):pass
