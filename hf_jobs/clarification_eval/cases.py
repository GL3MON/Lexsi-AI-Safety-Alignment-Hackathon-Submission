def cases(system):
    rows=[]
    def add(name,history,expected=None,clarify=None):
        rows.append(dict(name=name,messages=[dict(role='system',content=system)]+history,expected_calls=[expected] if expected else [],clarification=clarify))
    for n,(first,last,zip_code,email,oid) in enumerate([
        ('Elara','Mercer','63118','elara.mercer47@fastmail.com','#W8361942'),
        ('Ivo','Kovacs','97214','ivo.kovacs+receipts@proton.me','#W5927368')]):
        uid=f'{first.lower()}_{last.lower()}_5931'
        request=f'Please retrieve the current shipping status of order {oid}. Do not cancel or modify anything.'
        def user(text):return dict(role='user',content=text)
        add(f'{n}_missing_identity',[user(request)],clarify='identity')
        add(f'{n}_missing_zip',[user(f'{request} I cannot remember my email. My first name is {first} and last name is {last}. Use name and ZIP to find my account, but I have not provided my ZIP yet.')],clarify='zip')
        add(f'{n}_missing_last_name',[user(f'{request} I cannot remember my email. My first name is {first} and ZIP is {zip_code}. I have not provided my last name yet.')],clarify='last_name')
        add(f'{n}_missing_first_name',[user(f'{request} I cannot remember my email. My last name is {last} and ZIP is {zip_code}. I have not provided my first name yet.')],clarify='first_name')
        intro=[user(f'{request} My email is {email}.')]
        lookup=call('find_user_id_by_email',{'email':email})
        failed=intro+[lookup,tool('find_user_id_by_email','Error: User not found')]
        add(f'{n}_failed_lookup_missing_alternative',failed,clarify='identity')
        add(f'{n}_partial_answer_still_missing_zip',[
            user(f'{request} I cannot remember my email.'),
            dict(role='assistant',content='Please provide your first name, last name, and ZIP code so I can locate your account.'),
            user(f'My first name is {first} and my last name is {last}.')],clarify='zip')
        add(f'{n}_complete_email',intro,{'name':'find_user_id_by_email','arguments':{'email':email}})
        add(f'{n}_complete_name_zip',[user(f'{request} I cannot remember my email. My first name is {first}, last name {last}, and ZIP {zip_code}.')],{'name':'find_user_id_by_name_zip','arguments':{'first_name':first,'last_name':last,'zip':zip_code}})
        auth=intro+[lookup,tool('find_user_id_by_email',uid)]
        add(f'{n}_authenticated_get_profile',auth,{'name':'get_user_details','arguments':{'user_id':uid}})
        profile=auth+[call('get_user_details',{'user_id':uid}),tool('get_user_details',dict(user_id=uid,email=email,name=dict(first_name=first,last_name=last),orders=[oid]))]
        add(f'{n}_profile_retrieve_explicit_status',profile,{'name':'get_order_details','arguments':{'order_id':oid}})
        revised=f'{first.lower()}.{last.lower()}19@outlook.com'
        correction=failed+[dict(role='assistant',content='That email was not found. Please check your account email.'),user(f'The correct email is {revised}. Please use it to find my account and retrieve the order status.')]
        add(f'{n}_email_correction',correction,{'name':'find_user_id_by_email','arguments':{'email':revised}})
        recovery=profile+[call('get_order_details',{'orderID':oid}),tool('__runtime__',{'error':"Unexpected argument 'orderID'; required argument is 'order_id'. Please correct the field name."})]
        add(f'{n}_wrong_key_recovery',recovery,{'name':'get_order_details','arguments':{'order_id':oid}})
    assert len(rows)==24
    return rows
