"""Reserve public external evaluation records; never add them to training."""
from collections import Counter
import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from scripts.build_public_corpus import Builder
from scripts.corpus_integrity import describe

ROOT=Path(__file__).resolve().parents[1]/'data'


def main():
    excluded=set(pd.read_csv(ROOT/'HMEG_275K_English_Hindi_RealSource.csv',usecols=['normalized_sha256']).normalized_sha256)
    detector=Builder(1000)
    rows=[];seen=set();rejected=Counter();failures=[]
    def add(prompt,label,source,revision,license,split,identifier,family,category):
        if not isinstance(prompt,str) or not prompt.strip():rejected['empty']+=1;return
        if len(prompt)>10000:rejected['serving_length_limit']+=1;return
        info=describe(prompt);key=info['normalized_sha256']
        if key in excluded:rejected['overlap_with_275K_corpus']+=1;return
        if key in seen:rejected['duplicate_external']+=1;return
        language=detector.language(prompt)
        if language not in ('en','hi'):rejected['language_filter']+=1;return
        seen.add(key)
        rows.append(dict(prompt=prompt,label=label,language=language,source_dataset=source,
            source_revision=revision,source_url='https://huggingface.co/datasets/'+source,
            license=license,source_split=split,source_record_id=str(identifier),
            attack_family=family,source_category=category,is_synthetic='unknown',research_split='external_test',**info))
    specifications=[('JailbreakBench/JBB-Behaviors','886acc352a31533ffbcf4ef22c744658688086fc','behaviors','benign',0,'mit'),
                    ('JailbreakBench/JBB-Behaviors','886acc352a31533ffbcf4ef22c744658688086fc','behaviors','harmful',1,'mit'),
                    ('CohereLabs/aya_redteaming','5a16fee03c19d681b4f3527b6a750e1ed4d434c7',None,'hindi',1,'apache-2.0')]
    for source,revision,config,split,label,license in specifications:
        try:
            for row in load_dataset(source,config,split=split,revision=revision,streaming=True):
                add(row.get('Goal',row.get('prompt')),label,source,revision,license,split,row.get('Index','unknown'),
                    'harmful_behavior_proxy' if label else 'benign',str(row.get('Category',row.get('harm_category','unknown'))))
        except Exception as exc:failures.append({'source':source,'split':split,'error_type':type(exc).__name__})
    target=ROOT/'public_external_eval.jsonl'
    target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    report={'rows':len(rows),'sources':dict(Counter(row['source_dataset'] for row in rows)),
        'labels':dict(Counter(row['label'] for row in rows)),'languages':dict(Counter(row['language'] for row in rows)),
        'rejected':dict(rejected),'source_failures':failures,
        'scope':'Separate source-labelled evaluation. Positive JBB/Aya examples are harmful-behavior proxies, not necessarily jailbreak instructions.',
        'limitations':['Exact normalized overlaps with the entire 275K corpus removed; semantic/embedded-behavior overlap remains possible.',
            'Report each source separately. Aya Hindi is positive-only; NotInject is a separately saved benign-only benchmark.'],
        'skipped':{'Qualifire':'Gated; no access configured. No alternate copy used to bypass access.',
                   'TrustAIRLab':'DAN source alias occurs in the training aggregation; source isolation not established.'}}
    (ROOT/'public_external_eval_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
