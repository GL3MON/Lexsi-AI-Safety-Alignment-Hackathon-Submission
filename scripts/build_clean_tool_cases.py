"""Twenty frozen simple cases; explicit manual hints; no teacher generation."""
import hashlib,json
from pathlib import Path
from tau2_grpo.glaive_prompts import system_prompt,check_system
from tau2_grpo.glaive_signals import score_decision


def tools():
    def tool(name,description,props):return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':{k:{'type':'string'} for k in props},'required':props,'additionalProperties':False}}}
    return [tool('find_user_id_by_email','Look up an account by its complete email address.',['email']),tool('find_user_id_by_name_zip','Look up an account by name and five-digit ZIP.',['first_name','last_name','zip']),tool('get_weather','Retrieve the weather for a supplied city.',['city']),tool('get_user_details','Retrieve account details by verified user ID.',['user_id'])]


def build():
    catalog=tools();cases=[]
    def add(id,category,messages,hint,call=None,field=None,aliases=None):
        ref={'tool_calls':[call]} if call else {'decision':'clarify','missing_field':field,'missing_field_aliases':aliases or [field]}
        expected='<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>' if call else 'Please provide your '+(aliases or [field])[0]+'.'
        ref['assistant']=expected
        case={'id':id,'category':category,'scenario':category,'tools':catalog,'prompt':[{'role':'system','content':system_prompt(catalog)}]+messages,'manual_hint':hint,'reward_reference':ref}
        check_system(case['prompt'][0]['content'],catalog)
        assert score_decision(expected,case)['reward']>=.94
        cases.append(case)
    def user(text):return {'role':'user','content':text}
    def call(name,**args):return {'name':name,'arguments':args}
    for i,name in enumerate(('Nerina','Tavish')):
        add('missing_email_'+str(i),'missing',[user(f'Please find my account. My name is {name}. I have not given you my email yet.')],'Ask for the customer’s email before calling an account lookup.',field='email')
    for i,local in enumerate(('mira','keira')):
        add('missing_domain_'+str(i),'missing',[user(f'Please find my account. My email begins with {local}, but I cannot remember its domain after the @ sign.')],'Ask for the complete email address, including its missing domain. Do not call a tool yet.',field='email',aliases=['email','domain'])
    for i,(first,last) in enumerate((('Amara','Menon'),('Darian','Varela'))):
        add('missing_zip_'+str(i),'missing',[user(f'Find my account by name and ZIP. First name {first}, last name {last}. I have not supplied my ZIP code.')],'Ask only for the missing ZIP code before looking up the account.',field='zip',aliases=['zip','postal code'])
    for i in range(2):
        add('missing_city_'+str(i),'missing',[user('Please check the weather. I have not told you which city yet.' if i==0 else 'I need a weather forecast, but have not provided my location.')],'Ask which city the customer wants the weather for; do not call a tool yet.',field='city',aliases=['city','location'])
    for i,email in enumerate(('mira@example.org','keira@example.net')):
        add('copy_email_'+str(i),'copy',[user(f'Find my account. My email is {email}.')],'Copy the complete email from the customer into the email lookup call.',call('find_user_id_by_email',email=email))
    for i,(first,last,zip_) in enumerate((('Amara','Menon','02139'),('Darian','Varela','04861'))):
        add('copy_zip_'+str(i),'copy',[user(f'Find my account by name and ZIP: first name {first}, last name {last}, ZIP {zip_}.')],'Call the name-and-ZIP lookup using the supplied values; keep the ZIP as a string with its leading zero.',call('find_user_id_by_name_zip',first_name=first,last_name=last,zip=zip_))
    auth=call('find_user_id_by_email',email='nerina@example.org')
    add('copy_tool_id','copy',[user('Get my account details. My email is nerina@example.org.'),{'role':'assistant','content':'<tool_call>'+json.dumps(auth)+'</tool_call>'},user('<tool_response>'+json.dumps({'name':auth['name'],'content':'nerina_042'})+'</tool_response>')],'Copy the user ID from the lookup result into get_user_details.',call('get_user_details',user_id='nerina_042'))
    add('copy_city','copy',[user('Check the weather for Boston.')],'Copy the supplied city into the weather tool call.',call('get_weather',city='Boston'))
    def correction(id,old,new,hint):
        opening='Use these details: '+json.dumps(old['arguments'],ensure_ascii=False)
        history=[user(opening),{'role':'assistant','content':'<tool_call>'+json.dumps(old,ensure_ascii=False)+'</tool_call>'},user('<tool_response>'+json.dumps({'name':old['name'],'content':{'error':'No matching result; no state changed.'}})+'</tool_response>'),{'role':'assistant','content':'Please check the details and provide any correction.'},user('Correction: use these details instead: '+json.dumps(new['arguments'],ensure_ascii=False))]
        add(id,'correction',history,hint,new)
    for i,(old,new) in enumerate((('mira@example.net','mira@example.org'),('keira7@example.org','keira07@example.org'))):
        correction('correct_email_'+str(i),call('find_user_id_by_email',email=old),call('find_user_id_by_email',email=new),'Use the email from the latest customer correction, not the failed earlier address.')
    for i,(old,new) in enumerate((('02138','02139'),('48061','04861'))):
        correction('correct_zip_'+str(i),call('find_user_id_by_name_zip',first_name='Amara',last_name='Menon',zip=old),call('find_user_id_by_name_zip',first_name='Amara',last_name='Menon',zip=new),'Use the latest corrected ZIP exactly, preserving any leading zero.')
    correction('correct_name',call('find_user_id_by_name_zip',first_name='Amara',last_name='Menon',zip='02139'),call('find_user_id_by_name_zip',first_name='Amara',last_name='Mehra',zip='02139'),'Use the corrected surname from the latest customer message; retain the supplied first name and ZIP.')
    correction('correct_city',call('get_weather',city='Boston'),call('get_weather',city='Austin'),'Use the corrected city in the latest customer message, not the earlier city.')
    for c in cases:c['manual_hint']=c['manual_hint'].replace('’', chr(39))
    assert len(cases)==20
    return cases


def main():
    out=Path('outputs/diagnostics/clean-manual-hint-baseline');out.mkdir(parents=True,exist_ok=True)
    data=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in build());(out/'cases.jsonl').write_text(data)
    manifest={'cases':20,'categories':{'missing':8,'copy':6,'correction':6},'sha256':hashlib.sha256(data.encode()).hexdigest(),'manual_hints':True,'source':'New controlled fixtures; not a benchmark score. No cases used for training.','planned_seeds':[42,43],'temperature':.7,'top_p':1.}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':main()
