"""Fresh, resumable five-expert OOF training. Candidate artifacts only; never publish.

Accepts a checksum-verified local English corpus, or an activated migration corpus. Reuses existing expert implementations and the pinned frozen encoder.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from database.mongodb import Database
from scripts.stage_public_corpus import migration_settings
from ml.experts import CLASSES
from ml.experts.semantic_encoder import SemanticEncoder, asset_dir
from ml.training_embeddings import cached_embeddings
from ml.oof import build_oof
from ml.ensemble.research_meta import ResearchMetaClassifier, KEYS
from ml.evaluation.metrics import summarize
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    pending.replace(path)


def measures(y, probability):
    result = summarize(y, (probability >= .5).astype(int), probability, [0.0])
    # Cached scores do not measure serving latency. Never report zero as measured latency.
    result.pop('average_inference_time'); result.pop('p95_inference_time')
    result['metrics']['macro_f1']=float(f1_score(y,probability>=.5,average='macro'))
    if len(set(y))==2:
        precision,recall,_=precision_recall_curve(y,probability)
        result['metrics']['pr_auc']=float(auc(recall,precision))
        result['metrics']['average_precision']=float(average_precision_score(y,probability))
    else:
        result['metrics']['pr_auc']=None
        result['metrics']['average_precision']=None
    if not np.any(np.asarray(y)==1): result['metrics']['recall']=None
    return result


def train(path, validated_summary=None):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    dataset_version='HMEG-275K-v1'
    if validated_summary:
        from scripts.preflight_public_import import load_rows
        from scripts.corpus_integrity import validate
        audit=json.loads(Path(validated_summary).read_text(encoding='utf-8'))
        if audit.get('sha256')!=digest or not audit.get('integrity',{}).get('passed'):
            raise ValueError('Local dataset audit/checksum mismatch.')
        verified=list(load_rows(path))
        if not validate(verified)['passed'] or any(row['language']!='en' for row in verified):
            raise ValueError('English corpus integrity failed.')
        dataset_version=audit['dataset_version']
        del verified
        gc.collect()
    else:
        database = Database(migration_settings())
        try:
            if not database.connected():
                raise ValueError('Migration cluster unavailable.')
            state = database.db.research_metadata.find_one({'key': 'public_corpus_staging'})
            if not state or state.get('status') != 'activated' or state.get('sha256') != digest:
                raise ValueError('Migration activation has not completed successfully.')
            record = database.db.research_metadata.find_one({'key': 'dataset'})
            if not record or record['value'].get('status') != 'ready' or record['value'].get('sha256') != digest:
                raise ValueError('Activate and verify this exact corpus in the migration cluster first.')
        finally:
            database.close()
    frame = pd.read_csv(path, usecols=['prompt','label','group_id','research_split','source_dataset','language','attack_family','is_short_prompt'], keep_default_na=False)
    if len(frame) != 275000 or frame.label.value_counts().to_dict() != {0:137500,1:137500}:
        raise ValueError('Unexpected corpus balance or size.')
    partitions = {name:np.flatnonzero(frame.research_split.to_numpy()==name) for name in ('train','validation','test')}
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        if set(frame.iloc[partitions[left]].group_id) & set(frame.iloc[partitions[right]].group_id):
            raise ValueError('Partition group leakage.')
    versions = {name:importlib.metadata.version(name) for name in ('scikit-learn','numpy','xgboost','onnxruntime','tokenizers','joblib')}
    code_hash = hashlib.sha256()
    for source in sorted((ROOT/'ml').rglob('*.py')):
        code_hash.update(source.relative_to(ROOT).as_posix().encode()); code_hash.update(source.read_bytes())
    code_hash.update(Path(__file__).read_bytes())
    identity = {'dataset_sha256':digest,'versions':versions,'code_sha256':code_hash.hexdigest(),'folds':5,'seed':42}
    version = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:32]
    folder = ROOT/'models'/'new'/version
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'oof').mkdir(exist_ok=True)
    manifest_path = folder/'model_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        **identity,'model_version':version,'dataset_version':dataset_version,'dataset_row_count':len(frame),
        'created_at':datetime.now(timezone.utc).isoformat(),'python_version':platform.python_version(),
        'used_split_counts':{k:len(v) for k,v in partitions.items()},'sources':sorted(frame.source_dataset.unique()),
        'embedding_model':'sentence-transformers/all-MiniLM-L6-v2','embedding_runtime':'pinned quantized ONNX',
        'sentence_transformers_version':'not used; equivalent frozen ONNX encoder',
        'semantic_encoder':json.loads((asset_dir()/'manifest.json').read_text()),
        'preprocessing':'Original prompt formatting retained; learned features fit on fold-training rows only.',
        'status':'preparing','promotion_status':'not_promoted'}
    def progress(stage):
        manifest['status']=stage; write_json(manifest_path,manifest); print(stage,flush=True)
    try:
        manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        manifest['git_worktree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (OSError,subprocess.SubprocessError):
        manifest['git_commit']='unavailable'
    prompts=frame.prompt.to_numpy(); labels=frame.label.to_numpy(); lengths=frame.prompt.str.len().to_numpy()
    train_ids=partitions['train']; train_y=labels[train_ids]
    progress('Caching frozen sentence embeddings for the disjoint partitions')
    vectors=cached_embeddings(SemanticEncoder(),prompts.tolist(),ROOT/'models'/'embedding_cache',(asset_dir()/'manifest.json').read_bytes())
    def fit_expert(key, ids):
        expert=CLASSES[key]()
        started=time.perf_counter()
        if key=='meaning':
            expert.model.fit(vectors[ids],labels[ids]); expert.training_time=time.perf_counter()-started
        else:
            expert.fit(prompts[ids],labels[ids])
        return expert
    def score(expert,key,ids):
        if key=='meaning':
            return expert.model.predict_proba(vectors[ids])[:,list(expert.model.classes_).index(1)]
        return expert.probabilities(prompts[ids])
    def fold_predictions(fold,fit_ids,heldout_ids):
        columns=[]
        for key in KEYS:
            target=folder/'oof'/f'fold_{fold}_{key}.npy'
            if target.exists():
                values=np.load(target,allow_pickle=False)
            else:
                progress(f'OOF fold {fold+1}/5: {key}')
                expert=fit_expert(key,train_ids[fit_ids])
                values=score(expert,key,train_ids[heldout_ids])
                temporary=target.with_suffix('.pending.npy');np.save(temporary,values);temporary.replace(target)
                del expert;gc.collect()
            if values.shape!=(len(heldout_ids),):
                raise ValueError('Invalid fold checkpoint.')
            columns.append(values)
        return np.column_stack(columns)
    oof,fold_ids=build_oof(train_y,frame.iloc[train_ids].group_id.to_numpy(),fold_predictions)
    np.savez_compressed(folder/'oof_predictions.npz',scores=oof,fold_ids=fold_ids,training_row_indices=train_ids)
    experts={}
    for key in KEYS:
        target=folder/(key+'_expert.joblib')
        if target.exists():experts[key]=joblib.load(target)
        else:
            progress('Training final '+key+' expert on all training rows')
            experts[key]=fit_expert(key,train_ids);joblib.dump(experts[key],target,compress=3)
    scores={}
    for split in ('validation','test'):
        ids=partitions[split]; columns=[]
        for key in KEYS:
            target=folder/(split+'_'+key+'.npy')
            if target.exists():values=np.load(target,allow_pickle=False)
            else:
                progress('Scoring '+split+' / '+key)
                values=score(experts[key],key,ids);np.save(target,values)
            if values.shape!=(len(ids),) or not np.isfinite(values).all():raise ValueError('Invalid score checkpoint.')
            columns.append(values)
        scores[split]=np.column_stack(columns)
    progress('Fitting meta classifiers on out-of-fold TRAINING predictions only')
    metas={kind:ResearchMetaClassifier(kind).fit(oof,train_y,lengths[train_ids]) for kind in ('logistic','xgboost')}
    val_ids=partitions['validation']; val_y=labels[val_ids]; val_scores=scores['validation']
    comparisons={}
    fusion_scores={'average':val_scores.mean(axis=1),'majority':(val_scores>=.5).mean(axis=1),
                   'weighted':np.average(val_scores,axis=1,weights=[.25,.2,.2,.15,.2])}
    fusion_scores.update({'meta_'+kind:model.probabilities(val_scores,lengths[val_ids]) for kind,model in metas.items()})
    for name,values in fusion_scores.items():comparisons[name]=measures(val_y,values)
    preferred=max(metas,key=lambda k:(comparisons['meta_'+k]['metrics']['f1'],-comparisons['meta_'+k]['metrics']['fpr']))
    # Select the learned meta model on validation, while reporting simpler alternatives honestly.
    fast=val_scores[:,:2].mean(axis=1); fused=fusion_scores['meta_'+preferred]; routes=[]
    for low,high in ((0.,1.),(.1,.9),(.2,.8),(.3,.7),(.4,.6)):
        uncertain=(fast>low)&(fast<high); values=np.where(uncertain,fused,fast)
        row={'low':low,'high':high,'final':.5,**measures(val_y,values),
             'layer2_routing_rate':float(uncertain.mean()),'average_experts_executed':float(2+3*uncertain.mean())}
        routes.append(row)
    chosen=max(routes,key=lambda r:(r['metrics']['f1'],-r['metrics']['fpr'],-r['average_experts_executed']))
    thresholds={key:chosen[key] for key in ('low','high','final')}
    manifest.update(preferred_meta=preferred,routing_thresholds=thresholds,
        model_size_bytes={key:(folder/(key+'_expert.joblib')).stat().st_size for key in KEYS},
        training_time_final_experts_seconds={key:float(getattr(expert,'training_time',0)) for key,expert in experts.items()},
        expert_model_types={key:expert.method for key,expert in experts.items()},
        feature_settings={key:str(expert.model.get_params(deep=False)) for key,expert in experts.items()},
        meta_features=metas[preferred].feature_names,available_experts=list(KEYS),
        selection_policy='Highest validation F1; ties prefer lower FPR then fewer experts. Test never used for selection.')
    bundle={'experts':experts,'metas':metas,'preferred_meta':preferred,'version':version,'training':manifest,
            'routing_thresholds':thresholds,'preserve_format':True}
    joblib.dump(bundle,folder/'hmeg.joblib',compress=3)
    joblib.dump(metas[preferred],folder/'meta_classifier.joblib',compress=3)
    write_json(folder/'validation_fusion_comparison.json',comparisons)
    write_json(folder/'router_threshold_results.json',routes)
    test_ids=partitions['test']; test_scores=scores['test']; test_y=labels[test_ids]
    expert_results={key:measures(test_y,test_scores[:,i]) for i,key in enumerate(KEYS)}
    test_fused=metas[preferred].probabilities(test_scores,lengths[test_ids]);test_fast=test_scores[:,:2].mean(axis=1)
    routed=(test_fast>chosen['low'])&(test_fast<chosen['high'])
    hmeg=measures(test_y,np.where(routed,test_fused,test_fast))
    hmeg.update(layer1_exit_rate=float((~routed).mean()),layer2_routing_rate=float(routed.mean()),average_experts_executed=float(2+3*routed.mean()))
    write_json(folder/'expert_results.json',expert_results);write_json(folder/'hmeg_results.json',hmeg)
    test_probability=np.where(routed,test_fused,test_fast)
    slices={}
    for column in ('language','attack_family','is_short_prompt'):
        values=frame.iloc[test_ids][column].to_numpy()
        slices[column]={str(value):{'samples':int(np.sum(values==value)),**measures(test_y[values==value],test_probability[values==value])} for value in np.unique(values)}
    write_json(folder/'test_slices.json',{'scope':'Internal held-out test slices; not source-held-out OOD results','slices':slices})
    progress('fitted_and_scored_candidate_only; latency, baseline, ablation, OOD and promotion checks pending')
    print('Candidate directory:',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('--validated-summary');args=parser.parse_args()
    train(args.csv,args.validated_summary)
