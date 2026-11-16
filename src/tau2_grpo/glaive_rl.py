"""AlignTune-owned, multi-turn clarification GRPO with auditable Glaive rewards."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from tau2_grpo.glaive_signals import score_decision
from tau2_grpo.rl_config import load_settings


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','collect','train']);p.add_argument('--config',default='configs/rl-glaive.json');p.add_argument('--scenario');p.add_argument('--case-id');a=p.parse_args()
    raw=json.loads(Path(a.config).read_text());dataset=raw.pop('dataset_path')
    # Reuse the validated policy/objective configuration without introducing
    # dataset fields into the retail environment's config schema.
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w',suffix='.json') as f:
        json.dump(raw,f);f.flush();cfg=load_settings(f.name)
    rows=[json.loads(s) for s in Path(dataset).open()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate decisions')
    for row in rows:
        if score_decision(row['reward_reference']['assistant'],row)['reward']<.94:raise ValueError('Invalid reference '+row['id'])
    from tau2_grpo.glaive_episode import episode_rows, rollout
    rows=episode_rows(rows)
    if a.scenario:rows=[r for r in rows if r['scenario']==a.scenario]
    if a.case_id:rows=[r for r in rows if r['id']==a.case_id]
    if not rows:raise ValueError('No episodes match scenario')
    if a.command=='check':print(json.dumps({'decisions':len(rows),'config_valid':True,'initial_adapter':cfg.initial_adapter}));return
    import torch
    import wandb
    from tau2_grpo.rl_policy import LocalPolicy
    from tau2_grpo.rl_objective import update_group
    from tau2_grpo.aligntune_runtime import make_owner
    torch.set_num_threads(8)
    run=Path(cfg.output_dir)/('run-'+uuid4().hex[:10]);run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({**raw,'dataset_path':dataset},indent=2))
    source_manifest=Path(dataset).parent/'manifest.json'
    provenance={'dataset_sha256':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),'episode_starts':len(rows),'source_manifest':json.loads(source_manifest.read_text()) if source_manifest.exists() else None,'manual_hints_in_training':False}
    (run/'data_manifest.json').write_text(json.dumps(provenance,indent=2))
    print('Artifacts:',run,flush=True)
    wb=wandb.init(project=cfg.wandb_project,mode=cfg.wandb_mode,dir=str(run),config={**raw,'dataset_path':dataset})
    from tau2_grpo.glaive_logging import COLUMNS,rollout_records
    samples=wandb.Table(columns=COLUMNS,log_mode="INCREMENTAL")
    success=False
    try:
        policy=LocalPolicy(cfg,trainable=a.command=='train')
        from tau2_grpo.glaive_prompts import clean_tokenizer_template
        clean_tokenizer_template(policy.tokenizer)
        (run/"template-used.jinja").write_text(policy.tokenizer.chat_template)
        optimizer=torch.optim.AdamW([v for v in policy.model.parameters() if v.requires_grad],lr=cfg.learning_rate) if a.command=='train' else None
        owner=None
        def execute():
            rng=random.Random(cfg.seed);order=[];updates=0;history=[]
            for step in range(cfg.train_steps if optimizer else 1):
                if step%len(rows)==0:order=rng.sample(rows,len(rows))
                row=order[step%len(rows)];episodes=[]
                group=run/f'group-{step:04d}';group.mkdir()
                for i in range(cfg.group_size):
                    episode=rollout(policy,cfg,row,cfg.seed+step*cfg.group_size+i,i,with_reference=optimizer is not None)
                    episodes.append(episode)
                    (group/f'episode-{i:02d}.json').write_text(json.dumps(episode,indent=2))
                    (group/f'episode-{i:02d}.txt').write_text('\n\n'.join(('TOOL RESPONSE' if m['content'].startswith('<tool_response>') else m['role'].upper())+': '+m['content'] for m in episode['messages'])+'\n\nREWARD: '+json.dumps(episode['reward_details']))
                if not all(e['complete'] for e in episodes):raise RuntimeError('Incomplete group; inspect saved episodes')
                sample_rows,rollout_metrics=rollout_records(episodes,step)
                for i,episode in enumerate(episodes):
                    (group/f'episode-{i:02d}.json').write_text(json.dumps(episode,indent=2))
                    with (group/f'episode-{i:02d}.txt').open('a') as f:f.write('\n\nADVANTAGE: '+str(episode['advantage']))
                for sample in sample_rows:samples.add_data(*sample)
                metrics=update_group(policy,optimizer,episodes,cfg) if optimizer else {}
                updates+=metrics.get('optimizer_steps',0)
                report={'group':step,'scenario':row['scenario'],'mean_agent_turns':sum(e['agent_turns'] for e in episodes)/len(episodes),'customer_reply_rate':sum(e['customer_replied'] for e in episodes)/len(episodes),'mean_reward':sum(e['reward'] for e in episodes)/len(episodes),**metrics,**rollout_metrics}
                for key in ('call_precision','argument_accuracy','tool_selection','copy_accuracy','schema_validity'):
                    report[key]=sum(e['reward_details'].get(key,0) for e in episodes)/len(episodes)
                history.append(report);(run/'groups.json').write_text(json.dumps(history,indent=2));wb.log({**report,'rollouts/samples':samples});print(json.dumps(report),flush=True)
                if owner:owner.step(updates,report)
                if optimizer and metrics.get('optimizer_steps'):
                    target=run/f'checkpoint-{policy.version}';policy.model.save_pretrained(target,selected_adapters=['policy']);policy.tokenizer.save_pretrained(target/'policy')
                    torch.save({'optimizer':optimizer.state_dict(),'policy_version':policy.version},target/'optimizer.pt')
            (run/'result.json').write_text(json.dumps({'groups':len(history),'optimizer_steps':updates}))
            return SimpleNamespace(global_step=updates)
        if optimizer:
            owner=make_owner(kind='glaive_verifiable_grpo',config=cfg,run_dir=run,model=policy.model,dataset=dataset,tokenizer=policy.tokenizer,execute=execute);owner.train()
        else:execute()
        success=True
    finally:wb.finish(exit_code=0 if success else 1)

if __name__=='__main__':main()
