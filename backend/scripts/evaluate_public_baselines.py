"""Fit standalone baselines on training only, select on validation, report test.

Run after train_public_corpus finishes. Results remain candidate-only and do not
update the live website or promote models. No latency is inferred from cached scores.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from scripts.train_public_corpus import measures, write_json

ROOT=Path(__file__).resolve().parents[1]


def latency(model, prompts):
    model.predict_proba([prompts[0]])
    times=[]
    for prompt in prompts:
        start=time.perf_counter();model.predict_proba([prompt]);times.append((time.perf_counter()-start)*1000)
    return {'average_inference_time':float(np.mean(times)), 'p50_inference_time':float(np.median(times)),
            'p95_inference_time':float(np.percentile(times,95)), 'timing_samples':len(times),
            'timing_scope':'Warm local CPU, one prompt per call, including feature extraction. Not hosted latency.'}


def evaluate(csv, candidate):
    candidate=Path(candidate).resolve()
    if (ROOT/'models'/'new').resolve() not in candidate.parents:
        raise ValueError('Use a candidate directory under backend/models/new.')
    manifest=json.loads((candidate/'model_manifest.json').read_text(encoding='utf-8'))
    if not (candidate/'hmeg.joblib').exists():raise ValueError('Complete candidate expert training first.')
    with Path(csv).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=manifest['dataset_sha256']:raise ValueError('Dataset checksum mismatch.')
    frame=pd.read_csv(csv,usecols=['prompt','label','group_id','research_split'],keep_default_na=False)
    train=frame[frame.research_split=='train'];val=frame[frame.research_split=='validation'];test=frame[frame.research_split=='test']
    for other in (val,test):
        if set(train.group_id)&set(other.group_id):raise ValueError('Group leakage detected.')
    calibration=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(train.prompt,train.label,train.group_id))
    for left,right in calibration:
        if set(train.iloc[left].group_id)&set(train.iloc[right].group_id):raise ValueError('Calibration leakage.')
    def pipeline(classifier):
        return make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000,dtype=np.float32),classifier)
    factories={
        'logistic_regression':lambda:pipeline(LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
        'linear_svm':lambda:CalibratedClassifierCV(pipeline(LinearSVC(class_weight='balanced',random_state=42,max_iter=5000)),cv=calibration,ensemble=False,n_jobs=1),
        'random_forest':lambda:pipeline(RandomForestClassifier(n_estimators=100,max_depth=18,min_samples_leaf=2,class_weight='balanced',n_jobs=2,random_state=42)),
        'xgboost':lambda:pipeline(XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.08,n_jobs=2,random_state=42,tree_method='hist',eval_metric='logloss'))}
    rng=np.random.default_rng(42)
    indices=np.concatenate([rng.choice(np.flatnonzero(test.label.to_numpy()==label),size=min(50,int((test.label==label).sum())),replace=False) for label in (0,1)])
    sample=test.iloc[indices].prompt.tolist(); results=[]
    for name,factory in factories.items():
        target=candidate/('baseline_'+name+'.joblib'); record_path=candidate/('baseline_'+name+'.json')
        if record_path.exists():
            previous=json.loads(record_path.read_text(encoding='utf-8'))
            if previous.get('status')=='measured' and target.exists():results.append(previous);continue
        print('Baseline training:',name,'rows:',len(train),flush=True)
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                model=factory();start=time.perf_counter();model.fit(train.prompt,train.label);training_time=time.perf_counter()-start
            positive=list(model.classes_).index(1)
            validation=model.predict_proba(val.prompt)[:,positive];probability=model.predict_proba(test.prompt)[:,positive]
            if not np.isfinite(probability).all():raise ValueError('Nonfinite probabilities.')
            joblib.dump(model,target,compress=3);np.save(candidate/('baseline_'+name+'_test.npy'),probability)
            record={'name':name,'model_type':'single','status':'measured','model_version':manifest['model_version'],
                'dataset_version':manifest['dataset_version'],'dataset_sha256':digest,'training_samples':len(train),
                'validation_samples':len(val),'test_samples':len(test),'features':'Word TF-IDF unigram/bigram, up to 20,000 features (not a sample cap)',
                'training_time':training_time,'hyperparameters':str(model.get_params(deep=False)),
                'model_size_bytes':target.stat().st_size,'validation':measures(val.label.to_numpy(),validation),
                'warnings':[str(item.message) for item in recorded],**measures(test.label.to_numpy(),probability),**latency(model,sample)}
            del model;gc.collect()
        except Exception as exc:
            record={'name':name,'model_type':'single','status':'Not Evaluated','error_type':type(exc).__name__,
                    'model_version':manifest['model_version'],'dataset_version':manifest['dataset_version']}
        results.append(record);write_json(record_path,record);write_json(candidate/'baseline_results.json',results)
    measured=[r for r in results if r['status']=='measured']
    best=max(measured,key=lambda r:(r['validation']['metrics']['f1'],-r['validation']['metrics']['fpr'])) if measured else None
    write_json(candidate/'baseline_selection.json',{'selected_by':'Validation F1 then lower FPR; no test selection',
        'best_single_guardrail':best['name'] if best else None,'measured_baselines':len(measured),'required_baselines':len(factories)})
    print('Baseline evaluation complete:',len(measured),'of',len(factories),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('candidate');args=parser.parse_args()
    evaluate(args.csv,args.candidate)
