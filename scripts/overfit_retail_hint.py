"""Small HINT overfit with matched hint-free before/after generations."""
import argparse,fcntl,json,os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace,asdict
from tau2_grpo.opsd_retail import read_config
from tau2_grpo.rl_rollout import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=12);p.add_argument('--lr',type=float,default=1e-4);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in (source/'hint_turns.jsonl').read_text().splitlines() if s.strip()]
    chosen=[]
    predicates=[lambda r:r['task_id']=='14' and 'cannot be cancelled' in r['hint'] and 'delivered' in r['original_action'],
                lambda r:r['task_id']=='14' and r['turn']==10 and 'list_all_product_types' in r['original_action']]
    for predicate in predicates:
        item=next((r for r in rows if predicate(r) and r['candidate_id'] not in {x['candidate_id'] for x in chosen}),None)
        if item is None:raise RuntimeError('Expected diagnostic case unavailable')
        chosen.append(item)
    atomic_json(out/'samples.json',chosen)
    cfg,_=read_config(source/'launch-config.json');cfg=replace(cfg,learning_rate=a.lr,temperature=.8)
    atomic_json(out/'config.json',{'policy':asdict(cfg),'steps':a.steps,'lr':a.lr,'source':str(source),'scope':'overfit on two selected training cases; no generalization claim'})
    with open(Path.cwd()/'outputs/.retail-training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        import torch,wandb
        from tau2_grpo.rl_policy import LocalPolicy
        from tau2_grpo.hint_distillation import turn_distillation_loss
        from tau2_grpo.aligntune_runtime import make_owner
        policy=LocalPolicy(cfg,trainable=True);policy.sync_reference()
        optimizer=torch.optim.AdamW([v for v in policy.model.parameters() if v.requires_grad],lr=a.lr)
        run=wandb.init(project='tau2-retail-opsd',mode='online',dir=str(out),name=out.name,config={'steps':a.steps,'lr':a.lr,'cases':len(chosen),'experiment':'hint-overfit'})
        logs=[];generations=[]
        def evaluate(stage):
            original=policy.cfg;policy.cfg=replace(cfg,temperature=0.)
            try:
                for item in chosen:
                    prepared={'prompt_ids':item['student_prompt_ids']}
                    generated=policy.generate(prepared,192,42)
                    record={'stage':stage,'candidate_id':item['candidate_id'],'diagnosis':item['diagnosis'],'hint':item['hint'],'original_action':item['original_action'],'output':generated['text'],'ended_with_eos':generated['ended_with_eos']}
                    generations.append(record);print(json.dumps(record),flush=True)
            finally:policy.cfg=original
            atomic_json(out/'generations.json',generations)
            (out/'conversations.txt').write_text('\n\n'.join('STAGE: '+x['stage']+'\nCASE: '+x['candidate_id']+'\nDIAGNOSIS: '+x['diagnosis']+'\nTEACHER HINT (not given during generation): '+x['hint']+'\nORIGINAL ACTION: '+x['original_action']+'\nHINT-FREE GENERATION: '+x['output']+'\nEOS: '+str(x['ended_with_eos']) for x in generations))
        def execute():
            evaluate('before')
            for step in range(1,a.steps+1):
                optimizer.zero_grad(set_to_none=True);total=0.
                for item in chosen:
                    values=turn_distillation_loss(policy,item['student_prompt_ids'],item['teacher_prompt_ids'],item['student_completion_ids'],grad=True,temperature=cfg.temperature)
                    loss=values.mean()/len(chosen)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                    loss.backward();total+=float(loss.detach())
                norm=float(torch.nn.utils.clip_grad_norm_([v for v in policy.model.parameters() if v.requires_grad],1.,error_if_nonfinite=True));optimizer.step();policy.version+=1
                metrics={'step':step,'loss':total,'grad_norm':norm};logs.append(metrics);atomic_json(out/'training.json',logs);run.log(metrics);print(json.dumps(metrics),flush=True)
                if step in {6,a.steps}:
                    policy.model.save_pretrained(out/f'checkpoint-{step}',selected_adapters=['policy']);policy.tokenizer.save_pretrained(out/f'checkpoint-{step}'/'policy');evaluate('after-'+str(step))
            atomic_json(out/'result.json',{'completed':True,'steps':a.steps,'initial_loss':logs[0]['loss'],'final_loss':logs[-1]['loss'],'generation_file':str(out/'conversations.txt')})
            return SimpleNamespace(global_step=a.steps)
        try:
            owner=make_owner(kind='retail_hint_overfit',config=cfg,run_dir=out,model=policy.model,dataset=chosen,tokenizer=policy.tokenizer,execute=execute);owner.train()
        finally:run.finish()
if __name__=='__main__':main()
