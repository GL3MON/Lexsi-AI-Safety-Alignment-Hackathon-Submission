"""Batched distillation updates, held-out evaluation, and bounded checkpoints."""
import json,math,random,shutil,time
from pathlib import Path
from tau2_grpo.rl_rollout import atomic_json


def prune_checkpoints(run,limit):
    checkpoints=[]
    for p in Path(run).glob('checkpoint-*'):
        suffix=p.name.removeprefix('checkpoint-')
        if p.is_dir() and suffix.isdigit():checkpoints.append((int(suffix),p))
    for _,p in sorted(checkpoints)[:-limit]:shutil.rmtree(p)


def evaluate_validation(policy,cfg,run,opsd,step,wb):
    from tau2_grpo.rl_config import task_ids
    from tau2_grpo.rl import make_episode
    from tau2_grpo.rl_reporting import episode_check_report
    import wandb
    directory=Path(run)/'evaluation'/f'step-{step:06d}';directory.mkdir(parents=True,exist_ok=True)
    results=[]
    table=wandb.Table(columns=['task_id','seed','reward','complete','actor_turns','checks','episode_file'])
    policy.model.eval()
    for index,task in enumerate(task_ids(cfg,'validation')):
        for draw in range(opsd['eval_rollouts_per_task']):
            seed=cfg.seed+100000+index*opsd['eval_rollouts_per_task']+draw
            file=directory/f'task-{task}-seed-{seed}.json'
            episode=make_episode(policy,cfg,task,seed,file,with_reference=False)
            report=episode_check_report(episode)
            row={'task_id':task,'seed':seed,'reward':episode.get('reward'),'complete':episode.get('complete'),
                'checks':report['check_counts'],'turn_counts':report['turn_counts'],'error':episode.get('error'),'episode_file':str(file)}
            results.append(row);atomic_json(directory/'progress.json',results)
            table.add_data(str(task),seed,row['reward'],row['complete'],report['turn_counts']['actor_turns'],json.dumps(report['checks']),str(file))
            print(json.dumps({'event':'validation_episode','optimizer_step':step,**row}),flush=True)
    valid=[r for r in results if r['complete'] and r['reward'] is not None]
    summary={'optimizer_step':step,'partition':'validation','episodes':len(results),'evaluated':len(valid),
             'infrastructure_errors':len(results)-len(valid),'successes':sum(r['reward']==1 for r in valid),
             'mean_reward':sum(r['reward'] for r in valid)/len(valid) if valid else None,'results':results}
    atomic_json(directory/'summary.json',summary)
    wb.log({'eval/optimizer_step':step,'eval/mean_reward':summary['mean_reward'],
            'eval/pass_rate':summary['successes']/len(valid) if valid else None,
            'eval/infrastructure_errors':summary['infrastructure_errors'],'eval/samples':table})
    return summary


def fit(policy,optimizer,cfg,opsd,run,recovery,preserve,wb,on_step=None,eval_fn=None):
    import torch
    from tau2_grpo.hint_distillation import turn_distillation_loss
    run=Path(run);rng=random.Random(cfg.seed);updates=0;history=[]
    hint_mode=opsd['mode']=='hint';accum=opsd['gradient_accumulation_steps']
    epochs=opsd['epochs_per_round'];last_eval=None
    recovery_turns=[('recovery',example) for row in recovery for example in (row.get('distillation_examples') or [row])]
    total_records=len(recovery_turns)+len(preserve)
    total_updates=epochs*math.ceil(total_records/accum)
    episode_count=len({x['episode_id'] for x in recovery}) if hint_mode else None
    for epoch in range(epochs):
        records=recovery_turns+[('preserve',x) for x in preserve];rng.shuffle(records)
        epoch_started=time.monotonic();epoch_loss=0.
        for batch_start in range(0,len(records),accum):
            batch=records[batch_start:batch_start+accum];optimizer.zero_grad(set_to_none=True);batch_loss=0.;jsds=[]
            for offset,(kind,item) in enumerate(batch):
                index=batch_start+offset
                if kind=='recovery':
                    student=item['student_prompt_ids'];teacher=item['teacher_prompt_ids'];tokens=item['student_completion_ids']
                    weight=opsd['recovery_weight']*item.get('episode_turn_weight',1.)
                else:
                    student=item['prompt_ids'];teacher=student;tokens=item['completion_ids'];weight=opsd['preservation_weight']
                progress={'stage':'forward','epoch':epoch+1,'epochs':epochs,'turn':index+1,'total_turns':len(records),
                    'completed_turns':index,'optimizer_steps':updates,'total_optimizer_steps':total_updates,
                    'gradient_accumulation_steps':accum,'candidate_id':item.get('candidate_id')}
                atomic_json(run/'training-progress.json',progress);print(json.dumps({'event':'turn_started',**progress}),flush=True)
                values=turn_distillation_loss(policy,student,teacher,tokens,grad=True,k=100,temperature=cfg.temperature)
                # Unbiased estimate of the per-episode mean when sampling turns.
                normalizer=len(batch)*episode_count/len(records) if hint_mode else len(batch)
                loss=weight*values.mean()/normalizer
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite distillation loss')
                progress['stage']='backward';atomic_json(run/'training-progress.json',progress)
                loss.backward();batch_loss+=float(loss.detach());jsds.append(float(values.detach().mean()))
                elapsed=time.monotonic()-epoch_started
                progress.update(stage='turn_complete',completed_turns=index+1,turn_jsd=jsds[-1],
                    epoch_eta_seconds=elapsed/(index+1)*(len(records)-index-1))
                atomic_json(run/'training-progress.json',progress);print(json.dumps({'event':'turn_complete',**progress}),flush=True)
                wb.log({'progress/turns_completed':index+1,'progress/total_turns':len(records),
                        'progress/epoch':epoch+1,'progress/turn_jsd':jsds[-1],'progress/optimizer_steps':updates})
            norm=float(torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],cfg.max_grad_norm,error_if_nonfinite=True))
            optimizer.step();policy.version+=1;updates+=1;epoch_loss+=batch_loss
            metrics={'epoch':epoch+1,'optimizer_steps':updates,'total_optimizer_steps':total_updates,
                'loss':batch_loss,'mean_turn_jsd':sum(jsds)/len(jsds),'grad_norm':norm,'batch_turns':len(batch)}
            history.append(metrics);atomic_json(run/'training.json',history);wb.log(metrics);print(json.dumps({'event':'optimizer_update',**metrics}),flush=True)
            if on_step:on_step(updates,metrics)
            final=updates==total_updates;eval_due=updates%opsd['eval_steps']==0 or final
            save_due=updates%opsd['save_steps']==0 or eval_due or final
            if save_due:
                checkpoint=run/f'checkpoint-{updates}'
                policy.model.save_pretrained(str(checkpoint),selected_adapters=['policy']);policy.tokenizer.save_pretrained(checkpoint/'policy')
                torch.save({'optimizer':optimizer.state_dict(),'policy_version':policy.version,'epoch':epoch,
                    'next_turn':batch_start+len(batch),'torch_rng':torch.get_rng_state()},checkpoint/'optimizer.pt')
                prune_checkpoints(run,opsd['save_total_limit'])
            if eval_due:
                atomic_json(run/'training-progress.json',{'stage':'evaluation','optimizer_steps':updates,'total_optimizer_steps':total_updates})
                summary=(eval_fn or evaluate_validation)(policy,cfg,run,opsd,updates,wb)
                last_eval=updates;atomic_json(run/'latest-evaluation.json',summary)
    atomic_json(run/'training-progress.json',{'stage':'complete','optimizer_steps':updates,'total_optimizer_steps':total_updates,'last_eval_step':last_eval})
    return {'global_step':updates}
