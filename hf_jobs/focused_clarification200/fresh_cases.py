def cases(system):
    rows=[]
    def add(name,history,expected=None,kind=None):
        rows.append(dict(name=name,messages=[dict(role='system',content=system)]+history,expected_calls=[expected] if expected else [],clarification=kind))
    people=[('Safiya','Petrenko','04852','safiya.petrenko64@mail.example.net','#W6184732'),('Luca','Benali','29416','luca_benali+shop@example.org','#W3758196'),('Thora','Sundberg','83726','thora-sundberg91@example.com','#W9472531')]
    for n,(first,last,zip_code,email,oid) in enumerate(people):
        uid=f'customer_{first.lower()}_8642'
        request=f"I'd like the current status of order {oid}. Please leave the order unchanged."
        def user(text):return dict(role='user',content=text)
        add(f'{n}_missing_identity',[user(request)],kind='identity')
        add(f'{n}_missing_zip',[user(request+f' I cannot recall my email. My given name is {first} and my surname is {last}.')],kind='zip')
        add(f'{n}_missing_last_name',[user(request+f' I cannot recall my email. My first name is {first}; my ZIP is {zip_code}.')],kind='last_name')
        add(f'{n}_missing_first_name',[user(request+f' I cannot recall my email. My surname is {last}; my ZIP is {zip_code}.')],kind='first_name')
        intro=[user(request+f' Start by finding my account with this email: {email}.')]
        lookup=call('find_user_id_by_email',{'email':email})
        failed=intro+[lookup,tool('find_user_id_by_email','Error: User not found')]
        add(f'{n}_failed_lookup_missing_alternative',failed,kind='identity')
        add(f'{n}_partial_answer_still_missing_zip',[user(request+' I cannot remember my email.'),dict(role='assistant',content='What are your first name, last name, and ZIP code?'),user(f'{first} {last} is my full name.')],kind='zip')
        add(f'{n}_complete_email',intro,dict(name='find_user_id_by_email',arguments={'email':email}))
        add(f'{n}_complete_name_zip',[user(request+f' Please find my account by name and ZIP: first name {first}, last name {last}, ZIP {zip_code}. I cannot remember my email.')],dict(name='find_user_id_by_name_zip',arguments=dict(first_name=first,last_name=last,zip=zip_code)))
        auth=intro+[lookup,tool('find_user_id_by_email',uid)]
        add(f'{n}_authenticated_get_profile',auth,dict(name='get_user_details',arguments={'user_id':uid}))
        profile=auth+[call('get_user_details',{'user_id':uid}),tool('get_user_details',dict(user_id=uid,email=email,name=dict(first_name=first,last_name=last),orders=[oid]))]
        add(f'{n}_profile_retrieve_explicit_status',profile,dict(name='get_order_details',arguments={'order_id':oid}))
        corrected=f'{first.lower()}.{last.lower()}72@example.org'
        add(f'{n}_email_correction',failed+[dict(role='assistant',content='That email did not match an account. What is the correct account email?'),user(f'Use {corrected} instead. Please look me up with that corrected email, then check the order status.')],dict(name='find_user_id_by_email',arguments={'email':corrected}))
        add(f'{n}_wrong_key_recovery',profile+[call('get_order_details',{'orderID':oid}),tool('__runtime__',{'error':"Required argument 'order_id' missing; argument 'orderID' is unsupported. Retry using the required field name."})],dict(name='get_order_details',arguments={'order_id':oid}))
    assert len(rows)==36
    return rows
