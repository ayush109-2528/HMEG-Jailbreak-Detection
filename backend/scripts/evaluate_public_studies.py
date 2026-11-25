"""Flat fusion, controlled component ablations, and source-specific external tests."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from ml.ensemble.hmeg import HMEG, KEYS
from ml.ensemble.router import Router
from ml.ensemble.research_meta import ResearchMetaClassifier
from scripts.train_public_corpus import measures, write_json
from scripts.compare_public_candidate import evaluate

ROOT=Path(__file__).resolve().parents[1]


class OmittedExpertMeta:
    requires_prompt_length=True
    def __init__(self, model, omitted): self.model,self.omitted=model,omitted
    def score(self, scores, *, prompt_length):
        # Fixed-shape meta input, not a fabricated prediction from an executed expert.
        values={key:(0.0 if key==self.omitted else scores[key]) for key in KEYS}
        return self.model.score(values,prompt_length=prompt_length)


def timed(engine, prompts, **options):
    engine.predict(prompts[0],**options)
    results=[engine.predict(prompt,**options) for prompt in prompts]
    values=[r['total_latency_ms'] for r in results]
    return {'average_inference_time':float(np.mean(values)),'p50_inference_time':float(np.median(values)),
        'p95_inference_time':float(np.percentile(values,95)),'latency_samples':len(values),
        'timing_scope':'Warm local CPU; sampled actual engine calls; not hosted latency.'}


def run(csv, candidate):
    candidate=Path(candidate).resolve()
    if (ROOT/'models'/'new').resolve() not in candidate.parents:raise ValueError('Invalid candidate directory.')
    bundle=joblib.load(candidate/'hmeg.joblib');manifest=bundle['training'];route=bundle['routing_thresholds']
    with Path(csv).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['dataset_sha256']:raise ValueError('Dataset checksum mismatch.')
    frame=pd.read_csv(csv,usecols=['prompt','label','research_split'],keep_default_na=False)
    train=frame[frame.research_split=='train'];test=frame[frame.research_split=='test']
    scores=np.column_stack([np.load(candidate/('test_'+key+'.npy'),allow_pickle=False) for key in KEYS])
    if len(scores)!=len(test):raise ValueError('Score/test shape mismatch.')
    lengths=test.prompt.str.len().to_numpy(); y=test.label.to_numpy()
    rng=np.random.default_rng(42)
    indices=np.concatenate([rng.choice(np.flatnonzero(y==label),min(50,int(np.sum(y==label))),replace=False) for label in (0,1)])
    prompts=test.iloc[indices].prompt.tolist();flat=[]
    for method in ('majority','average','weighted','meta_logistic','meta_xgboost'):
        meta=bundle['metas']['logistic' if method=='meta_logistic' else 'xgboost']
        engine=HMEG(bundle['experts'],meta,Router(route['low'],route['high']),route['final'],bundle['version'],preserve_format=True)
        if method=='majority':p=(scores>=.5).mean(axis=1)
        elif method=='average':p=scores.mean(axis=1)
        elif method=='weighted':p=np.average(scores,axis=1,weights=[.25,.2,.2,.15,.2])
        else:p=meta.probabilities(scores,lengths)
        flat.append({'name':'Flat '+method,'model_type':'flat','model_version':bundle['version'],
            'dataset_version':manifest['dataset_version'],'evaluation_mode':'Full held-out batch scores; latency separately sampled using actual calls',
            **measures(y,p),**timed(engine,prompts,flat=True,fusion=method)})
    write_json(candidate/'flat_results.json',flat)
    oof=np.load(candidate/'oof_predictions.npz',allow_pickle=False)['scores']
    ablations=[]
    for omitted in KEYS:
        column=list(KEYS).index(omitted);masked=oof.copy();masked[:,column]=0
        model=ResearchMetaClassifier(bundle['preferred_meta']).fit(masked,train.label.to_numpy(),train.prompt.str.len().to_numpy())
        masked_test=scores.copy();masked_test[:,column]=0
        available=[i for i in (0,1) if i!=column]
        fast=scores[:,available].mean(axis=1);routed=(fast>route['low'])&(fast<route['high'])
        probability=np.where(routed,model.probabilities(masked_test,lengths),fast)
        experts={key:value for key,value in bundle['experts'].items() if key!=omitted}
        engine=HMEG(experts,OmittedExpertMeta(model,omitted),Router(route['low'],route['high']),route['final'],bundle['version'],preserve_format=True)
        ablations.append({'name':'HMEG without '+omitted,'model_type':'ablation','model_version':bundle['version'],
            'method':'Refit selected meta type on masked training OOF scores; omitted input fixed at zero; router renormalizes remaining fast experts. Full-model thresholds held fixed.',
            'evaluation_mode':'Full held-out batch scores; actual sampled calls for latency',
            'layer2_routing_rate':float(routed.mean()),'layer1_exit_rate':float((~routed).mean()),
            'average_experts_executed':float(len(available)+(4-len(available))*routed.mean()),
            **measures(y,probability),**timed(engine,prompts)})
    selected='meta_'+bundle['preferred_meta']
    without_router=next(row for row in flat if row['name']=='Flat '+selected)
    ablations.append({**without_router,'name':'HMEG without router','model_type':'ablation','layer1_exit_rate':0.0,'layer2_routing_rate':1.0,'average_experts_executed':5.0})
    write_json(candidate/'ablation_results.json',ablations)
    records=[]
    for filename in ('NotInject_external.jsonl','public_external_eval.jsonl'):
        records.extend(json.loads(line) for line in (ROOT/'data'/filename).read_text(encoding='utf-8').splitlines())
    selection=json.loads((candidate/'baseline_selection.json').read_text(encoding='utf-8'))
    best=selection['best_single_guardrail']
    baseline=joblib.load(candidate/('baseline_'+best+'.joblib')) if best else None
    engine=HMEG(bundle['experts'],bundle['metas'][bundle['preferred_meta']],Router(route['low'],route['high']),route['final'],bundle['version'],preserve_format=True)
    ood=[]
    for source in sorted({r['source_dataset'] for r in records}):
        rows=[r for r in records if r['source_dataset']==source];texts=[r['prompt'] for r in rows];labels=np.array([r['label'] for r in rows])
        safe_name=source.replace('/','_')
        for flat_mode in (False,True):
            result=evaluate(engine,rows,candidate/'ood_inference'/(safe_name+('_flat' if flat_mode else '_hmeg')),flat=flat_mode)
            ood.append({'name':'Flat ensemble' if flat_mode else 'HMEG','model_type':'ood','source':source,**result})
        for key,expert in bundle['experts'].items():
            result=measures(labels,expert.probabilities(texts))
            if len(set(labels))<2:result['metrics']['f1']=None
            ood.append({'name':key+' Expert','model_type':'ood','source':source,'samples':len(rows),'evaluation_mode':'Batched standalone expert',**result})
        if baseline:
            values=baseline.predict_proba(texts)[:,list(baseline.classes_).index(1)]
            result=measures(labels,values)
            if len(set(labels))<2:result['metrics']['f1']=None
            ood.append({'name':best,'model_type':'ood','source':source,'samples':len(rows),'selection':'Best baseline on validation, not external test',**result})
        write_json(candidate/'ood_results.json',{'source_specific_results':ood,
            'scope':'Source-specific evaluation only; JBB/Aya positives are harmful-behavior proxies. No pooled F1 is claimed.',
            'limitations':'Exact corpus overlaps removed during preparation; semantic/embedded-behavior overlap remains possible.'})
    print('Flat, ablation and external-source studies completed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('candidate');args=parser.parse_args()
    run(args.csv,args.candidate)
