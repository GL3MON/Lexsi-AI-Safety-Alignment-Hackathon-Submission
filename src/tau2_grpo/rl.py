"""Retail Tau2 collection, paired evaluation, and exact-context GRPO training."""
import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from tau2_grpo.rl_config import ROOT, load_settings, path, task_ids
from tau2_grpo.rl_rollout import atomic_json, collect_episode


def make_episode(policy,cfg,task_id,seed,output,with_reference=False):
    if cfg.environment in ('short_retail','short_bank'):
        from tau2_grpo.rl_short import ShortRetailEnvironment
        from tau2_grpo.rl_context import ContextController
        def no_summary(*args,**kwargs):raise RuntimeError('Short scenario exceeded the fixed context budget')
        controller=ContextController(policy.tokenizer,no_summary,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
        if cfg.environment=='short_bank':
            from tau2_grpo.rl_bank import bank_environment
            env=bank_environment(cfg,task_id)
        else:env=ShortRetailEnvironment(task_id,seed,cfg)
        try:return collect_episode(policy,env,controller,task_id,seed,cfg,output,with_reference)
        finally:env.close()
    from openai import OpenAI
    from transformers import AutoTokenizer
    from tau2_grpo.rl_context import ContextController,FrozenSummarizer
    from tau2_grpo.rl_tau2 import ExternalChat,RetailEnvironment
    external_tokenizer=AutoTokenizer.from_pretrained(cfg.external_tokenizer,revision=cfg.external_revision)
    client=OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=cfg.external_timeout_seconds,max_retries=0)
    external=ExternalChat(client,external_tokenizer,cfg,[])
    summarizer=FrozenSummarizer(client,cfg.user_model,external_tokenizer,cfg.external_context_limit,cfg.summary_max_tokens)
    controller=ContextController(policy.tokenizer,summarizer,cfg.context_limit,cfg.response_tokens,cfg.prompt_budget)
    env=RetailEnvironment(task_id,seed,cfg,external)
    try:
        return collect_episode(policy,env,controller,task_id,seed,cfg,output,with_reference)
    finally:
        client.close()


def check(cfg,actor='retail_sft'):
    from transformers import AutoConfig,AutoTokenizer
    from openai import OpenAI
    ids=task_ids(cfg)
    model_path=str(path(cfg.model_id)) if path(cfg.model_id).exists() else cfg.model_id
    model=AutoConfig.from_pretrained(model_path)
    if cfg.context_limit>model.max_position_embeddings: raise ValueError('Configured context exceeds student capacity')
    if actor=='retail_sft' and not (path(cfg.initial_adapter)/'adapter_config.json').is_file(): raise ValueError('SFT adapter does not exist yet')
    available={'data':[],'not_required_for_short_retail':cfg.environment in ('short_retail','short_bank')}
    if cfg.environment=='tau2':
        with OpenAI(base_url=cfg.user_base_url,api_key='local',timeout=15,max_retries=0) as client:
            available=client.models.list().model_dump()
            if cfg.user_model not in {m['id'] for m in available['data']}: raise ValueError('Frozen endpoint model is not available')
    tokenizer=AutoTokenizer.from_pretrained(str(path(cfg.initial_adapter)) if actor=='retail_sft' and (path(cfg.initial_adapter)/'tokenizer_config.json').is_file() else model_path)
    if cfg.chat_template_file:tokenizer.chat_template=path(cfg.chat_template_file).read_text()
    if cfg.system_prompt_file:
        import re
        catalogs=re.findall(r'<tools>(.*?)</tools>',path(cfg.system_prompt_file).read_text(),re.S)
        if len(catalogs)!=1 or not isinstance(json.loads(catalogs[0]),list):raise ValueError('Invalid configured system tool catalog')
    result=dict(ready=True,rl_tasks=len(ids),context_limit=cfg.context_limit,prompt_budget=cfg.prompt_budget,
        response_reserve=cfg.response_tokens,actor=actor,
        initial_adapter=str(path(cfg.initial_adapter)) if actor=='retail_sft' else None,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        system_prompt_file=cfg.system_prompt_file,
        user_prompt_file=cfg.user_prompt_file,
        endpoint_models=available,training_started=False)
    print(json.dumps(result,indent=2));return result


def main(argv=None):
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','collect','compare','train'])
    parser.add_argument('--config',default=str(ROOT/'configs/rl-retail.json'))
    parser.add_argument('--task-id')
    parser.add_argument('--actor',choices=['retail_sft','function_calling_base'],default='retail_sft')
    parser.add_argument('--partition',choices=['rl','validation','test'],default='rl')
    parser.add_argument('--output')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=JSON')
    args=parser.parse_args(argv)
    overrides={}
    for item in args.set:
        key,value=item.split('=',1);overrides[key]=json.loads(value)
    cfg=load_settings(args.config,overrides)
    if args.actor=='function_calling_base' and args.command not in ('collect','check'):
        parser.error('--actor function_calling_base is supported for inference collection only')
    if args.command=='train' and (cfg.temperature==0 or cfg.group_size<2 or cfg.top_p!=1.0):
        parser.error('GRPO training requires positive temperature, top_p=1 and group_size >= 2; greedy/nucleus configurations are inference only')
    if args.command=='check': check(cfg,args.actor);return
    if args.command=='train' and args.partition!='rl': parser.error('Training may use only the reserved RL partition')
    allowed=task_ids(cfg,args.partition)
    if args.task_id and args.task_id not in allowed: parser.error('Task is outside the selected partition')
    ids=[args.task_id] if args.task_id else allowed
    run=Path(args.output).resolve() if args.output else path(cfg.output_dir)/f'{args.command}-{uuid4().hex[:10]}'
    run.mkdir(parents=True,exist_ok=False)
    atomic_json(run/'config.json',asdict(cfg)|dict(command=args.command,partition=args.partition,task_ids=ids,actor=args.actor))
    print('RL artifacts: '+str(run),flush=True)
    # Only training/collection load weights. check is CPU-only metadata validation.
    import torch
    import wandb
    from tau2_grpo.rl_policy import LocalPolicy
    from tau2_grpo.rl_objective import update_group
    if cfg.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Expose exactly one allocated GPU using CUDA_VISIBLE_DEVICES')
    if os.environ.get('WANDB_DISABLED','').lower() in ('true','1','yes'):
        raise ValueError('Unset WANDB_DISABLED; this harness requires logging')
    wb=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,name=run.name,dir=str(run),
                  config=asdict(cfg)|dict(command=args.command,partition=args.partition,actor=args.actor))
    if wb is None or wb.disabled: raise RuntimeError('W&B initialization failed')
    if os.environ.get('RL_CHECKPOINT_DIR'):
        persistent=Path(os.environ['RL_CHECKPOINT_DIR'])/run.name
        atomic_json(persistent/'runtime.json',dict(wandb_url=wb.url,wandb_run_id=wb.id,initial_adapter=cfg.initial_adapter))
    success=False
    try:
        atomic_json(run/'runtime.json',dict(wandb_url=wb.url,wandb_run_id=wb.id,
            actor='paired' if args.command=='compare' else args.actor,
            retail_adapter_loaded=None if args.command=='compare' else args.actor=='retail_sft',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            framework='aligntune' if args.command=='train' else 'inference',
            algorithm='turn-aware clipped GRPO' if args.command=='train' else 'Inference-only evaluation; no optimizer updates',
            generation_config=dict(do_sample=cfg.temperature>0,temperature=cfg.temperature,
                top_p=cfg.top_p,top_k=0,repetition_penalty=1.0,max_new_tokens=cfg.response_tokens),
            orchestration='AlignTune turn-aware Tau2 extension; exact-context PyTorch/PEFT replay' if args.command=='train' else 'Tau2 inference harness',
            split_manifest_sha256=hashlib.sha256(path(cfg.split_manifest).read_bytes()).hexdigest(),
            summarizer_prompt_sha256=hashlib.sha256((ROOT/'templates/retail_memory_summarizer.txt').read_bytes()).hexdigest()))
        if args.command=='compare':
            policy=LocalPolicy(cfg,adapter=False)
            results=[]
            for label in ('function_calling_base','retail_sft'):
                if label=='retail_sft':
                    from peft import PeftModel
                    policy.model=PeftModel.from_pretrained(policy.model,str(path(cfg.initial_adapter)),adapter_name='policy',is_trainable=False)
                    policy.model.eval()
                episode=make_episode(policy,cfg,ids[0],cfg.seed,run/(label+'.json'))
                row=dict(model=label,task_id=ids[0],reward=episode['reward'],complete=episode['complete'],
                    turns=len(episode['actions']),summaries=len(episode['summaries']),
                    generated_tokens=sum(len(a['completion_ids']) for a in episode['actions']),
                    max_prompt_tokens=max((len(a['prompt_ids']) for a in episode['actions']),default=0),
                    stop_reason=episode.get('stop_reason'),error=episode.get('error'))
                results.append(row);print(json.dumps(row),flush=True)
                wb.log({label+'/reward':episode['reward'],label+'/summaries':len(episode['summaries'])})
            atomic_json(run/'comparison.json',dict(results=results,note='One paired diagnostic, not a benchmark; same task/seed/customer/summarizer and generation settings.'))
            from tau2_grpo.rl_reporting import write_comparison_checks
            write_comparison_checks(run)
            success=all(r['complete'] for r in results)
        else:
            policy=LocalPolicy(cfg,adapter=args.actor=='retail_sft',trainable=args.command=='train')
            atomic_json(run/'prompt-format.json',dict(
                chat_template_sha256=hashlib.sha256(policy.tokenizer.chat_template.encode()).hexdigest(),
                system_prompt_sha256=hashlib.sha256(path(cfg.system_prompt_file).read_bytes()).hexdigest() if cfg.system_prompt_file else None,
                tokenizer_source=str(path(cfg.initial_adapter)) if args.actor=='retail_sft' and (path(cfg.initial_adapter)/'tokenizer_config.json').is_file() else cfg.model_id,
                eos_token_id=policy.tokenizer.eos_token_id,pad_token_id=policy.tokenizer.pad_token_id,
                sft_template_enforced=args.command=='train'))
            def short_evaluation(label):
                if cfg.environment=='tau2' and cfg.tau2_eval_task_ids:
                    from tau2_grpo.rl_post20_eval import InferencePolicy
                    selected=cfg.tau2_eval_task_ids.split(',');assert set(selected)<=set(task_ids(cfg,'validation'))
                    original=policy.cfg;results=[]
                    try:
                        greedy=replace(cfg,temperature=0.,assistance_level='none');policy.cfg=greedy
                        directory=(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name if os.environ.get('RL_CHECKPOINT_DIR') else run)/label.replace('short-','tau2-')
                        for tid in selected:
                            episode=make_episode(InferencePolicy(policy),greedy,tid,cfg.seed,directory/f'episode-{tid}.json')
                            results.append(dict(task_id=tid,success=episode.get('native_reward')==1,native_reward=episode.get('native_reward'),training_reward=episode.get('reward'),complete=episode['complete'],turns=len(episode['actions'])))
                            atomic_json(directory/'summary.json',dict(results=results,passed=sum(r['success'] for r in results),total=len(results),native_tau2_evaluation=True))
                            if not episode['complete']:raise RuntimeError('Native validation infrastructure failure: '+str(episode.get('error')))
                        wb.log({label+'/native_success_rate':sum(r['success'] for r in results)/len(results)})
                    finally:policy.cfg=original
                    return
                if cfg.environment not in ('short_retail','short_bank'):return
                evaluation=[]
                original=policy.cfg
                try:
                    if cfg.environment=='short_bank':
                        from tau2_grpo.rl_bank import validation_cases
                        pairs=[(t,'bank') for t in validation_cases(cfg)]
                    else:pairs=[(t,k) for t in task_ids(cfg,'validation')[:3] for k in ('email_lookup','missing_zip','order_status')]
                    for task_id,scenario in pairs:
                        greedy=replace(cfg,temperature=0.,short_scenario=scenario)
                        policy.cfg=greedy
                        directory=(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name if os.environ.get('RL_CHECKPOINT_DIR') else run)/label
                        episode=make_episode(policy,greedy,task_id,cfg.seed,directory/f'{task_id}-{scenario}.json')
                        evaluation.append(dict(task_id=task_id,scenario=scenario,success=episode.get('short_task_success'),complete=episode['complete'],turns=len(episode['actions'])))
                        print(json.dumps(dict(event='short_greedy_evaluation',phase=label,**evaluation[-1])),flush=True)
                    directory=(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name if os.environ.get('RL_CHECKPOINT_DIR') else run)/label
                    atomic_json(directory/'summary.json',dict(results=evaluation,passed=sum(bool(r['success']) for r in evaluation),total=len(evaluation),native_tau2_evaluation=False))
                    wb.log({label+'/success_rate':sum(bool(r['success']) for r in evaluation)/len(evaluation)})
                finally:policy.cfg=original
            if args.command=='train' and (cfg.environment in ('short_retail','short_bank') or cfg.tau2_eval_task_ids):short_evaluation('short-before')
            optimizer=torch.optim.AdamW((p for p in policy.model.parameters() if p.requires_grad),lr=cfg.learning_rate) if args.command=='train' else None
            owner=None
            def execute_training():
                rng=random.Random(cfg.seed);history=[];updates=0
                if cfg.rollout_engine=='vllm':
                    from tau2_grpo.rl_vllm import collect_group,sync_adapter
                    sync_adapter(policy,cfg,run)
                groups=cfg.train_steps if args.command=='train' else 1
                for group_index in range(groups):
                    if args.command=='train':
                        if group_index%len(ids)==0:
                            if cfg.curriculum_file:
                                from tau2_grpo.tau2_curriculum import curriculum_order
                                task_order=curriculum_order(cfg,ids)
                            elif cfg.environment=='short_bank':
                                from tau2_grpo.rl_bank import balanced_order
                                task_order=balanced_order(cfg,ids,rng)
                            else:task_order=rng.sample(ids,len(ids))
                        task_id=task_order[group_index%len(task_order)]
                    else:task_id=rng.choice(ids)
                    group_cfg=cfg
                    assistance={}
                    if cfg.assisted_adaptive:
                        from tau2_grpo.assisted_tau2 import select_assistance
                        assistance=select_assistance(history,group_index)
                        group_cfg=replace(cfg,assistance_level=assistance['assistance_level'])
                    if cfg.environment=='short_retail' and args.command=='train':
                        from tau2_grpo.rl_short import KINDS
                        group_cfg=replace(cfg,short_scenario=KINDS[group_index%len(KINDS)])
                    episode_directory=(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name/f'group-{group_index:04d}') if os.environ.get('RL_CHECKPOINT_DIR') else run/f'group-{group_index:04d}'
                    if cfg.rollout_engine=='vllm':
                        episodes=collect_group(policy,group_cfg,task_id,cfg.seed+group_index*cfg.group_size,episode_directory,with_reference=optimizer is not None)
                    elif cfg.rollout_workers>1:
                        from tau2_grpo.rl_transformers_workers import collect_group as collect_transformers_group
                        episodes=collect_transformers_group(policy,group_cfg,task_id,cfg.seed+group_index*cfg.group_size,
                            episode_directory,with_reference=optimizer is not None)
                    else:
                        episodes=[make_episode(policy,group_cfg,task_id,cfg.seed+group_index*cfg.group_size+i,
                            episode_directory/f'episode-{i:02d}.json',with_reference=optimizer is not None) for i in range(cfg.group_size)]
                    summary=dict(group=group_index,task_id=task_id,policy_version=policy.version,
                        rewards=[e['reward'] for e in episodes],infrastructure_errors=sum(not e['complete'] for e in episodes),
                        summaries=sum(len(e['summaries']) for e in episodes))
                    if cfg.environment in ('short_retail','short_bank'):
                        scenario=group_cfg.short_scenario
                        if cfg.environment=='short_bank':
                            from tau2_grpo.rl_bank import load_bank
                            from tau2_grpo.rl_midturn import MID_PREFIX
                            from tau2_grpo.rl_bank import case_family
                            scenario=case_family(cfg,task_id)
                        summary.update(short_scenario=scenario,short_successes=sum(bool(e.get('short_task_success')) for e in episodes),native_rewards_not_evaluated=True)
                    summary.update(assistance)
                    if cfg.curriculum_file:
                        curriculum=json.loads(path(cfg.curriculum_file).read_text())['schedule'][group_index]
                        summary.update(curriculum_stage=curriculum['stage'],curriculum_retention=curriculum['retention'],task_complexity=curriculum['complexity'])
                    if cfg.reward_mode=='additive_grounded':
                        scored=[e for e in episodes if e['complete']]
                        summary['native_rewards']=[e.get('native_reward') for e in episodes]
                        for key in ('tau2_sum','argument_bonus','incorrect_argument_penalty','consecutive_failure_penalty','max_failure_streak'):
                            summary['reward_'+key]=sum(e['reward_details'][key] for e in scored)/len(scored) if scored else None
                    if optimizer is None and cfg.environment not in ('short_retail','short_bank'):
                        from tau2_grpo.rl_reporting import write_pass_at_k_report
                        pass_report=write_pass_at_k_report(run,episodes,cfg.group_size)
                        summary['observed_pass_at_k']=pass_report['observed_pass_at_k']
                        summary['successful_episodes']=pass_report['successful_episodes']
                    if optimizer is not None and all(e['complete'] for e in episodes):
                        metrics=update_group(policy,optimizer,episodes,cfg);summary.update(metrics)
                        if metrics['optimizer_steps']:
                            updates+=metrics['optimizer_steps']
                            checkpoint=(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name/f'checkpoint-{policy.version}') if os.environ.get('RL_CHECKPOINT_DIR') else run/f'checkpoint-{policy.version}'
                            policy.model.save_pretrained(str(checkpoint),selected_adapters=['policy'])
                            policy.tokenizer.save_pretrained(str(checkpoint/'policy'))
                            torch.save(dict(optimizer=optimizer.state_dict(),policy_version=policy.version,
                                            torch_rng=torch.get_rng_state()),checkpoint/'optimizer.pt')
                            (checkpoint/'COMPLETE').write_text('complete\n')
                            if os.environ.get('RL_CHECKPOINT_DIR'):
                                import shutil
                                previous=sorted((p for p in checkpoint.parent.glob('checkpoint-*') if (p/'COMPLETE').exists()),key=lambda p:int(p.name.rsplit('-',1)[-1]))
                                for old_checkpoint in previous[:-2]:shutil.rmtree(old_checkpoint)
                            print(json.dumps(dict(event='rl_checkpoint_saved',path=str(checkpoint),policy_version=policy.version)),flush=True)
                            if cfg.rollout_engine=='vllm':sync_adapter(policy,cfg,run)
                    elif optimizer is not None: summary['skipped_incomplete_group']=True
                    summary['optimizer_steps_total']=updates
                    summary['mean_reward']=sum(e['reward'] for e in episodes)/len(episodes) if all(e['complete'] for e in episodes) else None
                    history.append(summary);atomic_json(run/'groups.json',history)
                    if os.environ.get('RL_CHECKPOINT_DIR'):
                        persistent=Path(os.environ['RL_CHECKPOINT_DIR'])/run.name
                        atomic_json(persistent/'groups.json',history)
                        atomic_json(persistent/'config.json',asdict(cfg))
                    wb.log({k:v for k,v in summary.items() if isinstance(v,(int,float,bool))})
                    from tau2_grpo.rl_reporting import rollout_log_rows
                    columns,rows=rollout_log_rows(episodes,group_index,summary.get('advantages'))
                    wb.log({'rollouts/samples':wandb.Table(columns=columns,data=rows),
                            'rollouts/group':group_index})
                    print(json.dumps(summary),flush=True)
                    if owner is not None: owner.step(updates,summary)
                    if (cfg.environment=='short_bank' or cfg.tau2_eval_task_ids) and (group_index+1)%cfg.eval_every_groups==0:short_evaluation(f'short-step-{group_index+1}')
                if args.command=='train' and (cfg.environment in ('short_retail','short_bank') or cfg.tau2_eval_task_ids):short_evaluation('short-after')
                atomic_json(run/'result.json',dict(groups=len(history),optimizer_steps=updates,trained=updates>0))
                if os.environ.get('RL_CHECKPOINT_DIR'):atomic_json(Path(os.environ['RL_CHECKPOINT_DIR'])/run.name/'result.json',dict(groups=len(history),optimizer_steps=updates,trained=updates>0))
            if optimizer is not None:
                from tau2_grpo.aligntune_runtime import make_owner
                owner=make_owner(kind='turn_aware_tau2_grpo',config=cfg,run_dir=run,model=policy.model,dataset=ids,tokenizer=policy.tokenizer,execute=execute_training)
                owner.train()
            else: execute_training()
            success=True
    finally:
        wb.finish(exit_code=0 if success else 1)
    if not success: raise SystemExit('Comparison contains infrastructure failures; inspect saved episodes')


if __name__=='__main__':main()
