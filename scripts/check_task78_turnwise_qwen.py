"""Connected episode with Qwen's current-turn hint appended like manual guidance."""
import copy,json,re
from pathlib import Path
from openai import OpenAI
import torch
from tau2_grpo.rl_config import RetailRLConfig
from tau2_grpo.rl_policy import LocalPolicy
from tau2_grpo.glaive_episode import episode_rows,tool_feedback
from tau2_grpo.glaive_prompts import clean_tokenizer_template
from tau2_grpo.glaive_signals import score_decision
from tau2_grpo.hint_annotations import validate_annotation


def main():
    torch.set_num_threads(8)
    uid='78a0f4fbf464e4b67629db47c8ee2cbfe77fba8dce55d420a8afbeea54294201:2:missing_required'
    rows=episode_rows([json.loads(s) for s in Path('outputs/glaive-rl-signals-v2/train.jsonl').open()]);row=next(r for r in rows if r['id']==uid)
    original=json.load(open('outputs/rl-glaive/run-55f242da3a/group-0008/episode-00.json'))
    client=OpenAI(base_url='http://127.0.0.1:8004/v1',api_key='local',timeout=120,max_retries=1)
    cfg=RetailRLConfig(initial_adapter='outputs/rl-glaive/run-55f242da3a/checkpoint-2/policy',device='cpu',temperature=.7,top_p=1.)
    policy=LocalPolicy(cfg);clean_tokenizer_template(policy.tokenizer)
    out=Path('outputs/diagnostics/clean-task78-turnwise-qwen-structured');out.mkdir(parents=True,exist_ok=True)
    messages=copy.deepcopy(row['prompt']);turns=[];supplied=False
    def save():
        (out/'episode.json').write_text(json.dumps({'checkpoint':cfg.initial_adapter,'seed':106,'temperature':.7,'top_p':1.,'device':'cpu','hinter':'MIG2 qwen3.5-4b','placement':'Direct guidance appended only to current latest user message, like the manual baseline.','messages':messages,'turns':turns,'note':'Qwen writes each turn hint from a verified compact current-stage summary; full failed rollout retained as the diagnostic source. Future customer values are excluded until the customer replies. No training or real API calls.'},indent=2,ensure_ascii=False))
        (out/'conversation.txt').write_text('CHECKPOINT: '+cfg.initial_adapter+'\n\n'+'\n\n'.join('TURN '+str(t['turn'])+'\nDIAGNOSIS: '+t['diagnosis']+'\nCONFIDENCE: '+t['confidence']+'\nQWEN HINT: '+t['hint']+'\nCUSTOMER / OBSERVATION: '+t['latest_user_message']+'\nASSISTANT: '+t['output']+'\nCHECK: '+json.dumps(t['scores']) for t in turns))
    for turn in (1,2):
        contract={'decision':'copy_available_value' if supplied else 'ask_missing_field','field':'text','source':'latest customer reply' if supplied else 'customer must supply missing field'}
        request=[{'role':'system','content':Path('templates/opsd_structured_hinter_system.txt').read_text()},{'role':'user','content':json.dumps({'authoritative_turn_contract':contract,'observed_prior_failure':'The first rollout attempted translate_text with an empty required text argument and received missing_required_information. No later failed assistant action was observed.','current_state':'Customer has supplied the text; it is now available in the latest reply.' if supplied else 'The customer has not supplied any text yet.','instruction':'Return diagnosis, confidence and hint. Diagnose the observed failure or current state briefly; put the direct next action only in hint. Do not quote target values.'})}]
        attempts=[]
        for attempt in range(3):
            response=client.chat.completions.create(model='qwen3.5-4b',messages=request,temperature=0,max_tokens=320,response_format={'type':'json_object'},extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            raw=response.choices[0].message.content
            try:
                annotation=validate_annotation(json.loads(raw),supplied)
                hint=annotation['hint'];valid=True;validation_error=None
            except (ValueError,TypeError,KeyError) as exc:
                valid=False;validation_error=str(exc)
            attempts.append({'raw':raw,'accepted':valid,'validation_error':validation_error})
            if valid:break
            request.extend([{'role':'assistant','content':raw},{'role':'user','content':('The customer has ALREADY provided the text in the current history. Write a Copy instruction to use that supplied text; do not ask or confirm.' if supplied else 'The text is missing. Write an Ask instruction addressed to the assistant, not the customer question itself.')+' Return exactly diagnosis (up to 40 words), confidence (high/medium/low) and hint (up to 24 words).'}])
        if not valid:raise ValueError('No stage-correct short hint')
        model_messages=copy.deepcopy(messages)
        model_messages[-1]['content']+='\n\nGuidance for your next response: '+hint+'\nCarry out this instruction now; do not discuss the guidance.'
        rendered=policy.tokenizer.apply_chat_template(model_messages,tokenize=False,add_generation_prompt=True);(out/f'turn-{turn}-rendered.txt').write_text(rendered)
        ids=policy.tokenizer.apply_chat_template(model_messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        output=policy.generate({'prompt_ids':ids},192,106+(turn-1)*100003)
        reference=row if not supplied else {**row,'reward_reference':row['final_reference']}
        scores=score_decision(output['text'],reference)
        turns.append({'turn':turn,'diagnosis':annotation['diagnosis'],'confidence':annotation['confidence'],'hint':hint,'latest_user_message':messages[-1]['content'],'output':output['text'],'scores':scores,'hinter_request':request,'hinter_attempts':attempts,'ended_with_eos':output['ended_with_eos']})
        messages.append({'role':'assistant','content':output['text']});save();print('TURN',turn,'HINT',hint,'OUTPUT',repr(output['text']),'SCORES',scores,flush=True)
        if not supplied:
            if scores.get('clarification_correct'):
                messages.append(copy.deepcopy(row['customer_reply']));supplied=True;save();continue
            feedback=tool_feedback(output['text'],reference,scores,information_missing=True)
        else:feedback=tool_feedback(output['text'],reference,scores)
        if feedback:messages.append(feedback)
        save();break
    correct=supplied and bool(turns[-1]['scores'].get('exact_call_set') and turns[-1]['scores'].get('schema_validity')==1)
    (out/'summary.json').write_text(json.dumps({'turns':len(turns),'clarified':supplied,'correct_final_call':correct,'annotations':[{k:t[k] for k in ('turn','diagnosis','confidence','hint')} for t in turns]},indent=2));print('CORRECT FINAL CALL',correct,flush=True)

if __name__=='__main__':main()
