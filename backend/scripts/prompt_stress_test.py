"""Diagnostic evaluation only. Never fits models, writes labels to a database, or deploys."""
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from app.config import settings
from app.prediction_policy import apply_prediction_policy
from ml.utils.model_info import ModelStore

def run():
    root=Path(__file__).resolve().parents[1]
    fixture=root/'tests/fixtures/prompt_stress_cases.json'
    cases=json.loads(fixture.read_text(encoding='utf8'))
    store=ModelStore(settings)
    if store.engine is None: raise RuntimeError(store.error or 'No model artifact')
    engine=store.engine
    rows=[]
    for i,case in enumerate(cases):
        raw=engine.predict(case['prompt'])
        result=apply_prediction_policy(case['prompt'],raw)
        # Additional standalone diagnostics are not counted as hierarchical execution.
        all_scores={key:float(expert.score(case['prompt'])) for key,expert in engine.experts.items()}
        assert all(np.isfinite(s) and 0<=s<=1 for s in all_scores.values())
        assert result['experts_executed']==sum(r['executed'] for r in result['experts'].values())
        assert all(r['score'] is None for r in result['experts'].values() if not r['executed'])
        if result['decision_layer']==1:
            assert not any(result['experts'][key]['executed'] for key in ['meaning','pattern','instruction'])
        rows.append({**case,'result':result,'standalone_scores':all_scores})
        if (i+1)%50==0: print(f"Completed {i+1}/{len(cases)}",flush=True)
    def summary(items):
        labelled=[r for r in items if r['expected_label'] is not None]
        benign=[r for r in labelled if r['expected_label']==0]
        attack=[r for r in labelled if r['expected_label']==1]
        return dict(total=len(items),benign=len(benign),attack=len(attack),
            false_positives=sum(r['result']['label']==1 for r in benign),
            false_negatives=sum(r['result']['label']==0 for r in attack),
            benign_abstentions=sum(r['result']['label'] is None for r in benign),
            attack_abstentions=sum(r['result']['label'] is None for r in attack))
    categories={key:summary([r for r in rows if r['category']==key]) for key in sorted({r['category'] for r in rows})}
    originals=[r for r in rows if r['variant']=='original']
    flips=[]
    for r in rows:
        base=next(x for x in originals if x['family']==r['family'])
        if r['variant']!='original' and r['result']['prediction']!=base['result']['prediction']:
            flips.append(dict(id=r['id'],original=base['result']['prediction'],variant=r['result']['prediction']))
    repeat_failures=[]
    for r in originals[:20]:
        again=engine.predict(r['prompt'])
        first=r['result'].get('raw_model',r['result'])
        if abs(again['final_probability']-first['final_probability'])>1e-7 or again['label']!=first['label']:
            repeat_failures.append(r['id'])
    expert_errors={}
    for key in engine.experts:
        expert_errors[key]={name:sum(int(r['standalone_scores'][key]>=.5)!=label for r in originals if r['expected_label']==label)
                           for name,label in [('benign_false_positives',0),('attack_false_negatives',1)]}
    report=dict(model_version=engine.version,artifact_sha256=hashlib.sha256((settings.model_dir/'hmeg.joblib').read_bytes()).hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        scope='AI-authored diagnostics, not independently reviewed research benchmark. Variants are correlated; no training or deployment.',
        thresholds=dict(low=engine.router.low,high=engine.router.high,final=engine.threshold),
        overall=summary(rows),originals=summary(originals),categories=categories,
        variant_decision_flips=flips,repeat_failures=repeat_failures,expert_errors_on_originals=expert_errors,
        local_warm_latency_ms={'median':float(np.median([r['result']['total_latency_ms'] for r in rows[1:]])),
                               'p95':float(np.percentile([r['result']['total_latency_ms'] for r in rows[1:]],95))},
        rows=rows)
    output=root/'results/prompt-stress';output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# HMEG prompt stress test','',report['scope'],'',
        f"Model: {engine.version}",f"Original-only counts: {report['originals']}",
        f"All counts including correlated variants: {report['overall']}",'',
        '| Category | Benign | Attack | False positives | Misses | Abstentions |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in categories.items():
        lines.append(f"| {name} | {c['benign']} | {c['attack']} | {c['false_positives']} | {c['false_negatives']} | {c['benign_abstentions']+c['attack_abstentions']} |")
    lines+=['',f"Variant decision flips: {len(flips)}",f"Repeatability failures (20 prompts): {len(repeat_failures)}",'',
            '## Original prompt errors','']
    for r in originals:
        if r['expected_label'] is not None and r['result']['label']!=r['expected_label']:
            lines += [f"- **{r['id']}**: {r['prompt']}",
                f"  Expected {r['expected_label']}; observed {r['result']['prediction']}; score {r['result']['final_probability']}; layer {r['result']['decision_layer']}."]
    lines+=['','## Expert errors on originals','',json.dumps(expert_errors,indent=2),
        '', '## Interpretation','The broad positive training class also includes harmful requests. These controls focus on instruction manipulation. Diagnostics cannot establish population accuracy or verify model generalization. Do not use these already-inspected prompts as an untouched final test. Abstention is not a correct benign classification. All extra expert runs above are standalone diagnostics, not reported HMEG executions.']
    (output/'REPORT.md').write_text(chr(10).join(lines),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['overall','originals','categories','expert_errors_on_originals','repeat_failures']},indent=2))
    print('Variant flips:',len(flips))
    print('Report:',output/'REPORT.md')
if __name__=='__main__':run()
