"""Paired, resumable inference evaluation. Never deploys or mutates MongoDB."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from app.config import settings
from ml.ensemble.hmeg import HMEG
from ml.ensemble.router import Router
from ml.evaluation.metrics import summarize
from ml.utils.model_info import ModelStore
from scripts.corpus_integrity import normalized, sha
from scripts.train_public_corpus import write_json

ROOT=Path(__file__).resolve().parents[1]


def deployment_gate(current, candidate, old_notinject, new_notinject):
    return {
        'lower_test_fpr':candidate['fpr'] < current['fpr'],
        'recall_not_lower':candidate['recall'] >= current['recall'],
        'f1_not_lower':candidate['f1'] >= current['f1'],
        'notinject_fpr_not_worse':new_notinject['fpr'] <= old_notinject['fpr'],
    }


def english_ood_gate(current, candidate):
    """Fail closed on missing English external metrics or a regression."""
    checks={}
    for metric in ('fpr','recall','f1'):
        old,new=current.get(metric),candidate.get(metric)
        valid=old is not None and new is not None and np.isfinite(old) and np.isfinite(new)
        checks['english_ood_'+metric+'_not_worse']=bool(valid and (new<=old if metric=='fpr' else new>=old))
    return checks


def evaluate(engine, rows, directory, *, flat=False, fusion='meta_xgboost'):
    directory.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps({'version':engine.version,'flat':flat,'fusion':fusion,
        'router':[engine.router.low,engine.router.high,engine.threshold],
        'rows':[(r['normalized_sha256'],r['label']) for r in rows]},sort_keys=True).encode()).hexdigest()
    all_results=[]
    # Exclude cold-start initialization from warm inference measurements.
    if rows: engine.predict(rows[0]['prompt'],flat=flat,fusion=fusion)
    for start in range(0,len(rows),250):
        target=directory/f'chunk_{start:06d}.json'
        batch=rows[start:start+250]
        if target.exists():
            cached=json.loads(target.read_text(encoding='utf-8'))
            if cached['identity']!=identity or len(cached['rows'])!=len(batch):
                raise ValueError('Inference checkpoint belongs to a different comparison.')
            results=cached['rows']
        else:
            results=[]
            for row in batch:
                prediction=engine.predict(row['prompt'],flat=flat,fusion=fusion)
                probability=prediction['final_probability']
                if not np.isfinite(probability) or not 0<=probability<=1:
                    raise ValueError('Nonfinite model output.')
                results.append({'truth':row['label'],'label':prediction['label'],'probability':probability,
                    'latency_ms':prediction['total_latency_ms'],'experts_executed':prediction['experts_executed'],
                    'decision_layer':prediction['decision_layer'],'normalized_sha256':row['normalized_sha256']})
            write_json(target,{'identity':identity,'rows':results})
        all_results.extend(results)
        if (start+len(batch))%1000==0 or start+len(batch)==len(rows):
            print(directory.name,start+len(batch),'/',len(rows),flush=True)
    if not all_results:raise ValueError('No evaluation records remain after overlap filtering.')
    truth=[r['truth'] for r in all_results]; predictions=[r['label'] for r in all_results]
    timings=[r['latency_ms'] for r in all_results]
    report=summarize(truth,predictions,[r['probability'] for r in all_results],timings)
    report.update(samples=len(rows),class_counts=dict(Counter(truth)),model_version=engine.version,
        p50_inference_time=float(np.median(timings)),
        layer1_exit_rate=float(np.mean([r['decision_layer']==1 for r in all_results])),
        layer2_routing_rate=float(np.mean([r['decision_layer']==2 for r in all_results])),
        average_experts_executed=float(np.mean([r['experts_executed'] for r in all_results])),
        scope='Actual warm local CPU HMEG calls, before the serving abstention policy. Not hosted latency.')
    if len(set(truth))<2:
        report['metrics']['f1']=None
        if 1 not in truth:report['metrics']['recall']=None
    write_json(directory/'summary.json',report)
    return report


def compare(csv, candidate, old_csv):
    candidate=Path(candidate).resolve()
    if (ROOT/'models'/'new').resolve() not in candidate.parents:
        raise ValueError('Use a candidate under models/new.')
    manifest=json.loads((candidate/'model_manifest.json').read_text(encoding='utf-8'))
    if not manifest['status'].startswith('fitted_and_scored_candidate_only'):
        raise ValueError('Candidate training is not complete.')
    with Path(csv).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['dataset_sha256']:raise ValueError('Dataset mismatch.')
    frame=pd.read_csv(csv,usecols=['prompt','label','research_split','normalized_sha256','language','group_id'],keep_default_na=False)
    heldout=frame[frame.research_split=='test'].copy()
    old_hashes=set()
    for chunk in pd.read_csv(old_csv,usecols=['prompt'],chunksize=10000,keep_default_na=False):
        old_hashes.update(sha(normalized(prompt)) for prompt in chunk.prompt)
    before=len(heldout)
    heldout=heldout[~heldout.normalized_sha256.isin(old_hashes)].sample(frac=1,random_state=42)
    if set(heldout.label)!={0,1}:raise ValueError('Paired comparison needs both labels.')
    rows=heldout.to_dict('records')
    current_store=ModelStore(settings)
    if current_store.engine is None:raise ValueError('Current model cannot load.')
    current=current_store.engine
    bundle=joblib.load(candidate/'hmeg.joblib')
    if set(bundle['experts'])!={'word','format','meaning','pattern','instruction'}:
        raise ValueError('Candidate must contain all five experts.')
    route=bundle['routing_thresholds']
    proposed=HMEG(bundle['experts'],bundle['metas'][bundle['preferred_meta']],Router(route['low'],route['high']),
                  route['final'],bundle['version'],preserve_format=True)
    outputs=candidate/'paired_evaluation'
    audit={'new_test_rows':before,'exact_overlaps_with_old_dataset_removed':before-len(rows),'paired_rows':len(rows),
        'policy':'Exclude exact normalized overlap with ALL old dataset partitions, not only training.',
        'limitations':'This does not establish absence of semantic/template overlap with the old corpus.',
        'current_model_version':current.version,'candidate_model_version':proposed.version,'dataset_sha256':digest}
    write_json(candidate/'comparison_overlap_audit.json',audit)
    old_result=evaluate(current,rows,outputs/'current_test')
    new_result=evaluate(proposed,rows,outputs/'candidate_test')
    records=[json.loads(line) for line in (ROOT/'data'/'NotInject_external.jsonl').read_text(encoding='utf-8').splitlines()]
    candidate_hashes=set(frame.normalized_sha256)
    notinject=[r for r in records if r['normalized_sha256'] not in old_hashes and r['normalized_sha256'] not in candidate_hashes and r.get('language')=='en']
    old_ood=evaluate(current,notinject,outputs/'current_notinject')
    new_ood=evaluate(proposed,notinject,outputs/'candidate_notinject')
    checks=deployment_gate(old_result['metrics'],new_result['metrics'],old_ood['metrics'],new_ood['metrics'])
    english_external={}
    if manifest['dataset_version']=='HMEG-275K-English-v2':
        external=[json.loads(line) for line in (ROOT/'data'/'public_external_eval.jsonl').read_text(encoding='utf-8').splitlines()]
        excluded=old_hashes | set(frame.normalized_sha256)
        english=[r for r in external if r.get('language')=='en' and r['normalized_sha256'] not in excluded]
        if not english:raise ValueError('No independent English external records remain.')
        for source in sorted({r['source_dataset'] for r in english}):
            subset=[r for r in english if r['source_dataset']==source]
            if {r['label'] for r in subset}!={0,1}:raise ValueError('English external gate requires both labels.')
            safe=source.replace('/','_')
            previous=evaluate(current,subset,outputs/('current_english_'+safe))
            proposed_result=evaluate(proposed,subset,outputs/('candidate_english_'+safe))
            source_checks=english_ood_gate(previous['metrics'],proposed_result['metrics'])
            checks.update({safe+'_'+key:value for key,value in source_checks.items()})
            english_external[source]={'current':previous,'candidate':proposed_result,'checks':source_checks}
        write_json(candidate/'english_ood_comparison.json',{'sources':english_external,'scope':'English-only, exact overlaps with both corpora excluded. JBB positives are harmful-behavior proxies. Previously inspected challenge set; not a fresh independent confirmation benchmark.'})
    report={'candidate':new_result,'current':old_result,'candidate_notinject':new_ood,'current_notinject':old_ood,
        'checks':checks,'metric_gate_passed':all(checks.values()),'deployed':False,'english_external_sources':list(english_external),
        'remaining_gates':['Complete baseline/ablation/source-specific OOD reporting','Candidate artifact/API smoke and frontend checks','Verify deployed current version still matches comparison','Verify deployment configuration and rollback'],
        'note':'Passing metrics alone does not authorize skipping remaining checks. Failed candidates must not replace production.'}
    write_json(candidate/'promotion_comparison.json',report)
    print(json.dumps({'metric_gate_passed':report['metric_gate_passed'],'checks':checks,'deployed':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('candidate');parser.add_argument('old_csv');args=parser.parse_args()
    compare(args.csv,args.candidate,args.old_csv)
