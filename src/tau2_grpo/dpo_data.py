"""Exact rollout-format prompt and EOS-terminated completion encoding."""
def render_completion(tokenizer,messages,text):
 from tau2_grpo.rl_context import student_messages
 prompt=tokenizer.apply_chat_template(student_messages(messages),tokenize=True,add_generation_prompt=True,return_dict=False)
 full=tokenizer.apply_chat_template(student_messages(messages+[dict(role='assistant',content=text)]),tokenize=True,add_generation_prompt=False,return_dict=False)
 if full[:len(prompt)]!=prompt:raise ValueError('DPO template does not preserve exact prompt prefix')
 tail=full[len(prompt):]
 if tokenizer.eos_token_id not in tail:raise ValueError('Candidate lacks EOS')
 tail=tail[:tail.index(tokenizer.eos_token_id)+1]
 if len(prompt)+len(tail)>8192:raise ValueError('DPO candidate exceeds8K')
 return prompt,tail
