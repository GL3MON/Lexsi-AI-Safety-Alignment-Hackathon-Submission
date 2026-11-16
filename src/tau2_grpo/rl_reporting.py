"""Record each Tau2 check and explicit turn counts for paired policy replays."""

import csv
import json
from collections import Counter
from pathlib import Path


def episode_check_report(episode):
    info = (episode.get("simulation") or {}).get("reward_info") or {}
    basis = info.get("reward_basis") or []
    roles = Counter(message["role"] for message in episode.get("messages", []))
    actor = len(episode.get("actions", []))
    supplied=(episode.get("assistance") or {}).get("prefix_actions",0)
    checks = []

    def add(category, criterion, passed, details, description):
        checks.append({
            "category": category,
            "criterion": criterion,
            "description": description,
            "status": "pass" if passed is True else "fail" if passed is False else "not_evaluated",
            "passed": passed,
            "used_in_terminal_reward": category in basis,
            "details": details,
        })

    for check in info.get('short_checks') or []:
        add('SHORT_SUBGOAL',check['criterion'],check.get('passed'),check,'Short subgoal: '+check['criterion'])
    db = info.get("db_check")
    add("DB", "final_database_state", db.get("db_match") if db else None, db, "Final database state")
    for check in info.get("action_checks") or []:
        action = check["action"]
        arguments = action.get("arguments") or {}
        description = action["name"]
        if "order_id" in arguments:
            description += " / " + str(arguments["order_id"])
        add("ACTION", action, check.get("action_match"), check, description)
    for check in info.get("env_assertions") or []:
        add("ENV_ASSERTION", check["env_assertion"], check.get("met"), check, str(check["env_assertion"]))
    for check in info.get("nl_assertions") or []:
        add("NL_ASSERTION", check["nl_assertion"], check.get("met"), check, check["nl_assertion"])
    for check in info.get("communicate_checks") or []:
        add("COMMUNICATE", check["info"], check.get("met"), check, check["info"])
    return {
        "task_id": episode["task_id"],
        "reward": episode.get("native_reward", episode.get("reward")),
        "training_reward": episode.get("reward"),
        "reward_details": episode.get("reward_details"),
        "reward_basis": basis,
        "reward_breakdown": info.get("reward_breakdown"),
        "tau2_termination_reason": (episode.get("simulation") or {}).get("termination_reason"),
        "evaluation_note": (info.get("info") or {}).get("note"),
        "turn_counts": {
            "actor_turns": actor,
            "customer_turns": roles["user"],
            "total_dialogue_turns": actor + roles["user"] + supplied,
            "supplied_assistant_turns": supplied,
            "assistance_level": (episode.get("assistance") or {}).get("level","none"),
            "tool_response_messages": roles["tool"],
            "total_messages_excluding_system": sum(roles.values()) - roles["system"],
            "summary_events": len(episode.get("summaries", [])),
        },
        "turn_definition": "Actor turns count all generation attempts, including malformed calls. Total dialogue turns = actor + customer messages; tool responses and summaries are counted separately.",
        "check_categories": {
            key: "not_recorded" if info.get(key) is None else "no_checks" if info[key] == [] else "recorded"
            for key in ("db_check", "action_checks", "env_assertions", "nl_assertions", "communicate_checks")
        },
        "checks": checks,
        "check_counts": dict(Counter(check["status"] for check in checks)),
    }


def write_pass_at_k_report(run, episodes, k):
    """Observed pass@k on one task; unresolved infrastructure failures stay unknown."""
    run = Path(run)
    if len(episodes) != k or len({str(e['task_id']) for e in episodes}) != 1:
        raise ValueError('pass@k requires exactly k independent episodes of the same task')
    if len({e['episode_id'] for e in episodes}) != k:
        raise ValueError('Duplicate episode in pass@k samples')
    samples = []
    for i, episode in enumerate(episodes):
        report = episode_check_report(episode)
        evaluated = bool(episode.get('complete') and episode.get('reward') is not None)
        samples.append(dict(sample=i+1, episode_id=episode['episode_id'], seed=episode['seed'],
            evaluated=evaluated, success=report['reward'] == 1.0 if evaluated else None,
            reward=report['reward'], training_reward=episode.get('reward'), error=episode.get('error'), **{
                key: report[key] for key in ('turn_counts', 'check_counts', 'checks', 'reward_basis', 'tau2_termination_reason', 'evaluation_note')
            }))
    successes = sum(s['success'] is True for s in samples)
    evaluated = sum(s['evaluated'] for s in samples)
    passed = True if successes else False if evaluated == k else None
    config_path=run/'config.json'
    actor=json.loads(config_path.read_text()).get('actor','retail_sft') if config_path.exists() else 'retail_sft'
    result = dict(actor=actor,task_id=str(episodes[0]['task_id']), k=k, observed_pass_at_k=passed,
        successful_episodes=successes, evaluated_episodes=evaluated,
        infrastructure_errors=k-evaluated, samples=samples,
        note='One task, k independently seeded SFT rollouts. Observed pass@k means at least one reward-1 episode; this is not a multi-task benchmark estimate. Infrastructure failures remain unknown.')
    (run/'pass-at-k.json').write_text(json.dumps(result, indent=2)+'\n')
    text = [f"# {actor}: task {result['task_id']}, pass@{k}", '',
        f"Observed pass@{k}: {'PASS' if passed is True else 'FAIL' if passed is False else 'UNKNOWN (incomplete evaluation)'}",
        f"Successful episodes: {successes}/{k}; evaluated: {evaluated}/{k}", '', result['note'], '',
        '| Sample | Seed | Reward | Actor turns | Customer turns | Total dialogue turns | Summaries |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for sample in samples:
        t=sample['turn_counts']
        text.append(f"| {sample['sample']} | {sample['seed']} | {sample['reward']} | {t['actor_turns']} | {t['customer_turns']} | {t['total_dialogue_turns']} | {t['summary_events']} |")
    for sample in samples:
        text += ['', f"## Sample {sample['sample']}: individual checks", '']
        if sample['error']: text.append('Infrastructure error: '+sample['error'])
        if sample['tau2_termination_reason']: text.append('Tau2 termination: '+sample['tau2_termination_reason'])
        if sample['evaluation_note']: text.append(sample['evaluation_note'])
        for check in sample['checks']:
            gate='reward component' if check['used_in_terminal_reward'] else 'diagnostic'
            text.append(f"- {check['status'].upper()}: {check['description']} ({gate})")
    text += ['', 'Total dialogue turns = actor + customer; tool responses are recorded separately in the JSON.', '']
    body='\n'.join(text)
    (run/'pass-at-k.txt').write_text(body)
    (run/'pass-at-k.md').write_text(body)
    for file in run.glob('group-*/episode-*.json'):
        write_episode_conversation(file, json.loads(file.read_text()))
    return result


def write_episode_conversation(source, episode):
    """Human-readable exact outputs, with checks and causal summary boundaries."""
    source = Path(source)
    report = episode_check_report(episode)
    parts = [f"TASK: {episode['task_id']}\nSEED: {episode['seed']}\nREWARD: {episode.get('reward')}\nCOMPLETE: {episode.get('complete')}\nTAU2 TERMINATION: {report['tau2_termination_reason']}"]
    if report['evaluation_note']:
        parts.append(report['evaluation_note'])
    parts.append('\n'.join(f"{key}: {value}" for key, value in report['turn_counts'].items()))
    parts.append(report['turn_definition'])
    if episode.get('error'):
        parts.append('INFRASTRUCTURE ERROR: '+episode['error'])
    parts.append('INDIVIDUAL CHECKS:\n'+'\n'.join(
        f"{c['status'].upper()}: {c['description']} ({'reward component' if c['used_in_terminal_reward'] else 'diagnostic'})"
        for c in report['checks']))
    parts.append("Assistant text is the exact generated output. Tool messages are environment/runtime feedback. System policy/tools appear in the appendix.")
    actions=episode.get('actions', [])
    boundaries={a['raw_message_count']: a for i,a in enumerate(actions)
        if a.get('summary_count',0) > (actions[i-1].get('summary_count',0) if i else 0)}
    actor_turn=0
    for i,message in enumerate(episode.get('messages', [])):
        if message['role']=='system':
            continue
        if i in boundaries:
            a=boundaries[i]
            parts.append(f"=== CONTEXT SUMMARY BEFORE ACTOR TURN {a['turn']+1} ===\nNEXT PROMPT TOKENS: {len(a['prompt_ids'])}\nMEMORY (user role):\n"+a['messages'][1]['content'])
        label=message['role'].upper()
        if label=='ASSISTANT':
            actor_turn+=1
            label+=f' / ACTOR TURN {actor_turn}'
        if message.get('name'):
            label+=' / '+message['name']
        parts.append(f"--- MESSAGE {i}: {label} ---\n"+message['content'])
    for message in episode.get('messages', []):
        if message['role']=='system':
            parts.append('--- APPENDIX: SYSTEM POLICY / TOOLS ---\n'+message['content'])
    target=source.with_name(source.stem+'.conversation.txt')
    target.write_text('\n\n'.join(parts)+'\n')
    return target


def write_comparison_checks(run):
    """Backfill saved replays, and run automatically for future comparisons."""
    run = Path(run)
    labels = ("function_calling_base", "retail_sft")
    models = {}
    matrix = {}
    for label in labels:
        episode = json.loads((run / (label + ".json")).read_text())
        report = episode_check_report(episode)
        models[label] = report
        (run / (label + ".checks.json")).write_text(json.dumps(report, indent=2) + "\n")
        header = ["=== EVALUATION SUMMARY ===", f"Reward: {report['reward']}"]
        header.extend(f"{name}: {value}" for name, value in report["turn_counts"].items())
        header += [report["turn_definition"], "", "Individual check results:"]
        for check in report["checks"]:
            gate = "reward component" if check["used_in_terminal_reward"] else "diagnostic"
            header.append(f"{check['status'].upper()}: {check['description']} ({gate})")
        header += ["=== END EVALUATION SUMMARY ===", ""]
        for suffix in (".txt", ".conversation.txt"):
            file = run / (label + suffix)
            if file.exists():
                original = file.read_text()
                if original.startswith("=== EVALUATION SUMMARY ==="):
                    original = original.split("=== END EVALUATION SUMMARY ===", 1)[1].lstrip("\n")
                file.write_text("\n".join(header) + "\n" + original)
        for check in report["checks"]:
            key = check["category"] + ":" + json.dumps(check["criterion"], sort_keys=True)
            row = matrix.setdefault(key, {
                "category": check["category"], "criterion": check["criterion"],
                "description": check["description"], "results": {},
            })
            row["results"][label] = check
    output = {"models": models, "check_matrix": list(matrix.values())}
    (run / "comparison-checks.json").write_text(json.dumps(output, indent=2) + "\n")
    with (run / "comparison-checks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "description", "base_status", "sft_status", "base_used_in_reward", "sft_used_in_reward", "criterion_json"])
        for row in matrix.values():
            results = [row["results"].get(label, {}) for label in labels]
            writer.writerow([row["category"], row["description"],
                *[result.get("status", "not_recorded") for result in results],
                *[result.get("used_in_terminal_reward") for result in results],
                json.dumps(row["criterion"], sort_keys=True)])

    lines = ["# Base versus retail SFT: individual checks", "", "| Metric | Function-calling base | Retail SFT |", "| --- | ---: | ---: |"]
    lines.append(f"| Task reward | {models[labels[0]]['reward']} | {models[labels[1]]['reward']} |")
    for key, title in (("actor_turns", "Actor turns"), ("customer_turns", "Customer turns"),
                       ("total_dialogue_turns", "Total dialogue turns"), ("tool_response_messages", "Tool responses"),
                       ("total_messages_excluding_system", "All messages excluding system"), ("summary_events", "Summaries")):
        values = [models[label]["turn_counts"][key] for label in labels]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    lines += ["", models[labels[0]]["turn_definition"], "", "## Checks", "", "| Category | Check | Base | Retail SFT |", "| --- | --- | --- | --- |"]
    for row in matrix.values():
        statuses = [row["results"].get(label, {}).get("status", "not_recorded").upper() for label in labels]
        description = row["description"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['category']} | {description} | {statuses[0]} | {statuses[1]} |")
    lines += ["", "## Reward basis", ""]
    for label in labels:
        lines.append(f"- {label}: {', '.join(models[label]['reward_basis']) or 'not recorded'}.")
    lines += ["", "Action checks are diagnostic unless ACTION is in the reward basis. A passing diagnostic is not a passing task reward. Missing checks are never marked failed.",
              "Full criteria, expected action arguments, judge explanations, and category availability are retained in comparison-checks.json.", ""]
    text = "\n".join(lines)
    (run / "comparison-checks.md").write_text(text)
    (run / "comparison-checks.txt").write_text(text)

    # Preserve the original comparison summary while adding the explicit records.
    summary_path = run / "comparison.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for result in summary.get("results", []):
            report = models.get(result.get("model"))
            if report:
                result.update(turn_counts=report["turn_counts"], checks=report["checks"],
                              reward_basis=report["reward_basis"], check_counts=report["check_counts"])
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output
def rollout_log_rows(episodes, group_index, advantages=None):
    """Keep native outcome, shaped reward, and each rollout's advantage visible."""
    import json
    from tau2_grpo.rl_objective import group_advantages
    if advantages is None:
        advantages=(group_advantages(episodes)
                    if len(episodes)>=2 and all(e.get('complete') for e in episodes)
                    else [None]*len(episodes))
    if len(advantages)!=len(episodes):raise ValueError('Advantage count differs from rollouts')
    columns=['group','rollout','task_id','episode_id','policy_version','native_reward',
             'training_reward','advantage','actor_turns','unique_correct_calls',
             'max_failure_streak','stop_reason','checks','conversation','assistance_level','supplied_prefix_actions']
    rows=[]
    for index,(episode,advantage) in enumerate(zip(episodes,advantages,strict=True)):
        parts=episode.get('reward_details',{})
        checks=[{'category':c['category'],'description':c['description'],'status':c['status']}
                for c in episode.get('check_report',{}).get('checks',[])]
        conversation='\n\n'.join(m['role'].upper()+':\n'+m.get('content','')
                                   for m in episode.get('messages',[]) if m['role']!='system')
        rows.append([group_index,index,episode['task_id'],episode['episode_id'],
                     episode.get('policy_version'),episode.get('native_reward',episode.get('reward')),
                     episode.get('reward'),advantage,len(episode.get('actions',[])),
                     parts.get('unique_correct_calls'),parts.get('max_failure_streak'),
                     episode.get('stop_reason'),json.dumps(checks,ensure_ascii=False),conversation,(episode.get('assistance') or {}).get('level','none'),(episode.get('assistance') or {}).get('prefix_actions',0)])
    return columns,rows

