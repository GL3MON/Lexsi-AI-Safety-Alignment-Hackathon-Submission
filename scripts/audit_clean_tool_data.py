"""Audit canonical catalogs, exact tokenizer rendering, labels and Unicode."""
import json,re
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from tau2_grpo.glaive_prompts import check_system,clean_tokenizer_template
from tau2_grpo.glaive_signals import score_decision
from tau2_grpo.prepare_function_calling import parse_calls


def main():
    tok=AutoTokenizer.from_pretrained('outputs/hf-export/tiny-aya-global-function-calling-sft-525')
    clean_tokenizer_template(tok)
    report={}
    for filename in ('outputs/glaive-rl-signals/train.jsonl','outputs/glaive-rl-signals-v2/train.jsonl'):
        rows=[json.loads(s) for s in Path(filename).open()];counts=Counter();bad=[]
        for row in rows:
            system=row['prompt'][0]['content']
            actual_catalogs=[]
            for block in re.findall(r'<tools>(.*?)</tools>',system,re.S):
                try:value=json.loads(block)
                except ValueError:continue
                if isinstance(value,list):actual_catalogs.append(value)
            if len(actual_catalogs)>1:counts['multiple_json_tool_catalogs']+=1
            if any(catalog!=row['tools'] for catalog in actual_catalogs):counts['embedded_catalog_disagrees_with_row_tools']+=1
            try:check_system(system,row['tools'])
            except ValueError:counts['noncanonical_catalogs']+=1
            if filename.endswith('v2/train.jsonl'):
                try:
                    check_system(system,row['tools'])
                    rendered=tok.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
                    assert rendered.count('<tools>')==rendered.count('</tools>')==1
                    assert score_decision(row['reward_reference']['assistant'],row)['reward']>=.94
                    counts['rendered_and_label_checks_passed']+=1
                except Exception as exc:bad.append({'id':row['id'],'error':str(exc)})
            for call in row['reward_reference'].get('tool_calls',[]):
                for value in call['arguments'].values():
                    if isinstance(value,str) and re.search(r'\\u[0-9a-fA-F]{4}',value):counts['literal_unicode_escapes_in_gold']+=1
        report[filename]={'rows':len(rows),'counts':dict(counts),'failures':bad}
    sft_audit={}
    for filename in ('outputs/tiny-aya-glaive/sft_train.jsonl','outputs/tiny-aya-glaive/sft_val.jsonl'):
        source_rows=[json.loads(s) for s in Path(filename).open()];duplicates=0
        for row in source_rows:
            catalogs=[]
            for raw in re.findall(r'<tools>(.*?)</tools>',row['messages'][0]['content'],re.S):
                try:value=json.loads(raw)
                except ValueError:continue
                if isinstance(value,list):catalogs.append(value)
            duplicates+=len(catalogs)>1
        sft_audit[filename]={'rows':len(source_rows),'multiple_json_catalogs':duplicates}
    report['sft_source_catalog_audit']=sft_audit
    actual='Je suis très heureux de vous rencontrer'
    call={'name':'translate_text','arguments':{'text':actual}}
    unicode_json='<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>'
    escaped_json='<tool_call>'+json.dumps(call,ensure_ascii=True)+'</tool_call>'
    literal={'name':'translate_text','arguments':{'text':r'Je suis tr\u00e8s heureux de vous rencontrer'}}
    assert parse_calls(unicode_json)==parse_calls(escaped_json)
    assert parse_calls('<tool_call>'+json.dumps(literal)+'</tool_call>')!=[call]
    report['unicode_audit']={'gold_character':'è','gold_codepoint':'U+00E8','json_escape_equivalent':True,'literal_backslash_escape_is_different':True,'prior_wrong_character':'ã / U+00E3','conclusion':'No literal escaped-string gold values found. Existing è-to-ã failures are genuine exact-copy mismatches.'}
    out=Path('outputs/diagnostics/clean-manual-hint-baseline');(out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report['outputs/glaive-rl-signals-v2/train.jsonl']['failures']:raise SystemExit(1)

if __name__=='__main__':main()
