"""Audit actual supervised retail sources for call loops; never filters training data."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau2_grpo.prepare_function_calling import parse_calls

ROOT = Path(__file__).resolve().parents[1]


def signature(call):
    return json.dumps({'name': call['name'], 'arguments': call['arguments']}, sort_keys=True, ensure_ascii=False)


def calls_from(message):
    native = message.get('tool_calls') or []
    if native:
        result = []
        for item in native:
            function = item.get('function', item)
            args = function['arguments']
            result.append(dict(name=function['name'], arguments=json.loads(args) if isinstance(args, str) else args,
                               id=item.get('id')))
        return result
    return [dict(call, id=None) for call in parse_calls(message.get('content') or '')]


def result_for(messages, index, call):
    results = []
    for m in messages[index+1:]:
        if m['role'] != 'tool':
            break
        if call['id'] and m.get('tool_call_id') and m['tool_call_id'] != call['id']:
            continue
        content = m.get('content') or ''
        try:
            body = json.loads(content)
        except (ValueError, TypeError):
            body = content
        if isinstance(body, dict) and 'name' in body and 'content' in body:
            body = body['content']
        error = bool(m.get('error')) or (isinstance(body, str) and body.lstrip().lower().startswith('error')) or (isinstance(body, dict) and bool(body.get('error')))
        results.append(dict(content=body, explicit_error=error))
    return dict(results=results, explicit_error=any(r['explicit_error'] for r in results),
                has_result=bool(results), result_signature=json.dumps([r['content'] for r in results], sort_keys=True, ensure_ascii=False))


def connected(messages, first, second):
    between = messages[first['message_index']+1:second['message_index']]
    return second['assistant_turn'] == first['assistant_turn']+1 and bool(between) and all(m['role'] == 'tool' for m in between)


def analyze(messages, covered, max_period):
    calls = []
    assistant_turn = 0
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        assistant_turn += 1
        for call in calls_from(message):
            # Coverage refers to original message indices, not repeated SFT chunk context.
            if i not in covered:
                raise ValueError('Source tool action has no selected SFT target: '+str(i))
            calls.append(dict(name=call['name'], arguments=call['arguments'], signature=signature(call),
                              message_index=i, assistant_turn=assistant_turn,
                              readonly=call['name'].startswith(('find_', 'get_', 'list_')),
                              **result_for(messages, i, call)))
    segments = []
    for call in calls:
        if not segments or not connected(messages, segments[-1][-1], call):
            segments.append([])
        segments[-1].append(call)
    events = []
    for segment in segments:
        i = 0
        while i < len(segment):
            best = None
            for period in range(1, min(max_period, (len(segment)-i)//2)+1):
                pattern = [c['signature'] for c in segment[i:i+period]]
                repetitions = 1
                while i+(repetitions+1)*period <= len(segment) and [c['signature'] for c in segment[i+repetitions*period:i+(repetitions+1)*period]] == pattern:
                    repetitions += 1
                if repetitions >= 2 and (best is None or repetitions*period > best[0]):
                    best = (repetitions*period, period, repetitions)
            if best is None:
                i += 1
                continue
            length, period, repetitions = best
            block = segment[i:i+length]
            same_results = all(block[j]['result_signature'] == block[j % period]['result_signature'] for j in range(length))
            events.append(dict(kind='consecutive_identical_call' if period == 1 else 'repeated_call_cycle',
                               period=period, repetitions=repetitions, calls=block,
                               readonly=all(c['readonly'] for c in block),
                               any_explicit_error=any(c['explicit_error'] for c in block),
                               all_have_results=all(c['has_result'] for c in block), same_results=same_results))
            i += length
    # Look across dialogue too: repeated error -> ask -> bad ID -> error can
    # be a behavioral loop even when the argument changes on every attempt.
    known = {(e['calls'][0]['message_index'], e['calls'][-1]['message_index'], e['period']) for e in events}
    for i in range(len(calls)):
        for period in range(1, min(max_period, (len(calls)-i)//2)+1):
            pattern = [c['signature'] for c in calls[i:i+period]]
            repetitions = 1
            while i+(repetitions+1)*period <= len(calls) and [c['signature'] for c in calls[i+repetitions*period:i+(repetitions+1)*period]] == pattern:
                repetitions += 1
            if repetitions < 2:
                continue
            block = calls[i:i+repetitions*period]
            key = (block[0]['message_index'], block[-1]['message_index'], period)
            if key in known:
                continue
            # Strict cycles already cover their own subranges.
            if all(connected(messages, a, b) for a,b in zip(block, block[1:])):
                continue
            known.add(key)
            events.append(dict(kind='repeated_call_cycle_across_dialogue', period=period, repetitions=repetitions,
                calls=block, readonly=all(c['readonly'] for c in block),
                any_explicit_error=any(c['explicit_error'] for c in block),
                all_have_results=all(c['has_result'] for c in block),
                same_results=all(block[j]['result_signature'] == block[j % period]['result_signature'] for j in range(len(block)))))
    errors = []
    for call in calls+[None]:
        if call and call['explicit_error']:
            errors.append(call)
        else:
            if len(errors) >= 3:
                events.append(dict(kind='three_or_more_failed_calls_across_dialogue', period=None,
                    repetitions=len(errors), calls=list(errors), readonly=all(c['readonly'] for c in errors),
                    any_explicit_error=True, all_have_results=True,
                    same_results=len({c['result_signature'] for c in errors}) == 1))
            errors = []
    # Broader repeats include refreshes after mutations and rechecking after dialogue.
    # Preserve them for review without classifying all duplicate reads as loops.
    repeated = []
    previous = {}
    for n, call in enumerate(calls):
        old = previous.get(call['signature'])
        if old is not None:
            first = calls[old]
            between_calls = calls[old+1:n]
            raw_between = messages[first['message_index']+1:call['message_index']]
            repeated.append(dict(name=call['name'], arguments=call['arguments'],
                first_message=first['message_index'], repeated_message=call['message_index'],
                consecutive_tool_turns=connected(messages, first, call),
                same_results=first['result_signature'] == call['result_signature'],
                previous_result_error=first['explicit_error'],
                intervening_write=any(not c['readonly'] for c in between_calls),
                intervening_customer_messages=sum(m['role'] == 'user' for m in raw_between),
                intervening_tool_names=[c['name'] for c in between_calls]))
        previous[call['signature']] = n
    return calls, events, repeated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'outputs/retail-data/retail-sft-final/sft.jsonl')
    parser.add_argument('--max-cycle-length', type=int, default=8)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    sources = defaultdict(lambda: dict(rows=[], targets=set()))
    masked_context_calls = 0
    for line_number, line in enumerate(dataset.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        item = sources[row['source_episode']]
        item['rows'].append(dict(line=line_number, id=row['id']))
        item['targets'].update(row['source_target_indices'])
        for i,m in enumerate(row['messages']):
            if m['role'] == 'assistant' and i not in row['loss_message_indices']:
                masked_context_calls += len(parse_calls(m['content']))
    events, repeats, source_reports = [], [], []
    review = dataset.parent/'tool-call-loop-reviews'
    review.mkdir(exist_ok=True)
    for source, selected in sorted(sources.items()):
        episode = json.loads(Path(source).read_text())
        record = episode['record']
        calls, found, broader = analyze(episode['messages'], selected['targets'], args.max_cycle_length)
        metadata = dict(task_id=str(record['task_id']), source_episode=source, reward=record.get('reward'), sft_examples=selected['rows'])
        for event in found:
            event.update(metadata)
            event['event_id'] = len(events)+1
            start, end = event['calls'][0]['message_index'], event['calls'][-1]['message_index']
            excerpt = [f"Task: {record['task_id']} / source reward: {record.get('reward')}\nSource: {source}\nKind: {event['kind']} / period: {event['period']} / repetitions: {event['repetitions']}\nOriginal source message indices are zero-based.\n"]
            for i in range(max(0,start-3), min(len(episode['messages']),end+4)):
                m = episode['messages'][i]
                if m['role'] == 'system':
                    continue
                excerpt.append(f"MESSAGE {i} / {m['role'].upper()}\n"+json.dumps(m, indent=2, ensure_ascii=False))
            file = review/f"event-{event['event_id']:03d}-task-{record['task_id']}.txt"
            file.write_text('\n\n'.join(excerpt)+'\n')
            event['review_file'] = str(file)
            events.append(event)
        repeats.extend(dict(**metadata, **r) for r in broader)
        source_reports.append(dict(**metadata, tool_calls=len(calls), loop_candidates=len(found), broader_repeat_pairs=len(broader)))
    task_ids = sorted({e['task_id'] for e in events}, key=int)
    strong = [e for e in events if e['readonly'] and not e['any_explicit_error'] and e['all_have_results'] and e['same_results']]
    result = dict(dataset=str(dataset), dataset_sha256=before, dataset_modified=False,
        scope='Actual training source episodes; repeated masked chunk context is excluded. Validation/rejected sources are not included.',
        definitions=dict(loop_candidate='Identical tool-call sequences repeat at least twice (cycle lengths 1..'+str(args.max_cycle_length)+'), both with and without intervening dialogue. Also flags >=3 consecutive failed calls in the tool-call stream, allowing intervening dialogue and changed arguments.',
            caveat='Candidates require review; two repeated reads or error retries do not prove an infinite/behavioral loop. Broader duplicate calls may be legitimate refreshes.'),
        source_trajectories=len(sources), supervised_tool_calls=sum(s['tool_calls'] for s in source_reports),
        excluded_masked_context_tool_calls=masked_context_calls,
        loop_candidate_events=len(events), loop_candidate_source_trajectories=len({e['source_episode'] for e in events}),
        loop_candidate_task_ids=task_ids,
        unchanged_read_repeat_events=len(strong), broader_repeat_pairs=len(repeats),
        events=events, broader_repeats=repeats, sources=source_reports)
    (dataset.parent/'tool-call-loop-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (dataset.parent/'tool-call-loop-candidate-task-ids.txt').write_text('\n'.join(task_ids)+'\n')
    lines = ['# Retail SFT tool-call repetition audit', '',
        f"Training sources: {len(sources)}; actual supervised tool calls: {result['supervised_tool_calls']}.",
        f"Excluded repeated masked-context tool calls: {masked_context_calls}.",
        f"Loop candidates: {len(events)} events in {result['loop_candidate_source_trajectories']} trajectories, across {len(task_ids)} tasks.",
        f"Unchanged, non-error read-only repeat events: {len(strong)}.",
        f"Broader identical-call repeat pairs: {len(repeats)} (not automatically loops).", '',
        result['definitions']['loop_candidate'], result['definitions']['caveat'], '',
        '| Event | Task | Source reward | Kind | Calls per cycle | Repetitions | Any error | Same results | Tools |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for e in events:
        names=', '.join(c['name'] for c in e['calls'][:e['period'] or len(e['calls'])])
        lines.append(f"| {e['event_id']} | {e['task_id']} | {e['reward']} | {e['kind']} | {e['period']} | {e['repetitions']} | {e['any_explicit_error']} | {e['same_results']} | {names} |")
    lines += ['', '## Exact source excerpts', '']
    lines += [f"- Event {e['event_id']}, task {e['task_id']}: [review excerpt]({e['review_file']})" for e in events]
    lines += ['', 'All source paths, SFT row IDs, call arguments, paired results and broader repeats are retained in tool-call-loop-audit.json. No training data was changed.', '']
    text='\n'.join(lines)
    (dataset.parent/'tool-call-loop-audit.txt').write_text(text)
    (dataset.parent/'tool-call-loop-audit.md').write_text(text)
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == before
    print(json.dumps({key:result[key] for key in ['source_trajectories','supervised_tool_calls','excluded_masked_context_tool_calls','loop_candidate_events','loop_candidate_source_trajectories','loop_candidate_task_ids','unchanged_read_repeat_events','broader_repeat_pairs']}, indent=2))


if __name__ == '__main__':
    main()
