"""Reuse frozen features for unchanged text; train/evaluate a fresh English candidate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from ml.experts.semantic_encoder import SemanticEncoder,asset_dir
from scripts.train_public_corpus import train,write_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'

def key(prompts,identity):
    digest=hashlib.sha256(identity)
    for prompt in prompts:
        raw=prompt.encode('utf-8');digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()

def main():
    csv=DATA/'HMEG_275K_EnglishOnly_RealSource.csv'
    audit=DATA/'HMEG_275K_EnglishOnly_RealSource_summary.json'
    meta=json.loads(audit.read_text(encoding='utf-8'))
    with csv.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['sha256']
    original=DATA/'HMEG_275K_English_Hindi_RealSource.csv'
    with original.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==meta['original_sha256']
    old=pd.read_csv(original,usecols=['prompt'],keep_default_na=False).prompt.tolist()
    new=pd.read_csv(csv,usecols=['prompt'],keep_default_na=False).prompt.tolist()
    identity=(asset_dir()/'manifest.json').read_bytes()
    cache=ROOT/'models'/'embedding_cache';source=cache/key(old,identity);target=cache/key(new,identity)
    target.mkdir(exist_ok=True)
    if not (target/'progress.json').exists():
        assert json.loads((source/'progress.json').read_text())['completed']==len(old)
        previous=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
        assert previous.shape==(len(old),384)
        output=np.lib.format.open_memmap(target/'vectors.npy',mode='w+',dtype=np.float32,shape=(len(new),384))
        unchanged=np.array([a==b for a,b in zip(old,new)])
        assert len(old)==len(new) and int((~unchanged).sum())==616
        output[unchanged]=previous[unchanged]
        encoder=SemanticEncoder();indices=np.flatnonzero(~unchanged)
        output[indices]=encoder.encode([new[i] for i in indices]);output.flush()
        assert np.isfinite(output).all()
        write_json(target/'reuse_audit.json',{'original_dataset_sha256':meta['original_sha256'],'new_dataset_sha256':meta['sha256'],'encoder_identity_sha256':hashlib.sha256(identity).hexdigest(),'unchanged_prompt_vectors_reused':int(unchanged.sum()),'new_vectors_encoded':len(indices),'scope':'Frozen encoder features only; no fitted expert or meta classifier reused. Batch-dependent quantized numerical differences may occur; actual single-prompt inference is evaluated separately.'})
        write_json(target/'progress.json',{'completed':len(new)})
        del output,previous,encoder
    print('Frozen features ready. Training all five experts from scratch.',flush=True)
    del old,new
    train(csv,audit)
    candidates=[]
    for path in (ROOT/'models'/'new').glob('*/model_manifest.json'):
        m=json.loads(path.read_text(encoding='utf-8'))
        if m['dataset_sha256']==meta['sha256'] and m['status'].startswith('fitted_and_scored_candidate_only'):candidates.append(path.parent)
    if len(candidates)!=1:raise RuntimeError('Candidate identification is ambiguous')
    candidate=candidates[0]
    for stage,module,args in [('paired_comparison','scripts.compare_public_candidate',[str(csv),str(candidate),'C:/Users/ayush/Downloads/HMEG_300K_Augmented_Dataset.csv']),('single_guardrails','scripts.evaluate_public_baselines',[str(csv),str(candidate)]),('flat_ablation_ood','scripts.evaluate_public_studies',[str(csv),str(candidate)])]:
        write_json(candidate/'evaluation_progress.json',{'stage':stage,'deployed':False,'scope':'English-only candidate; Hindi external results out of supported scope'})
        print(stage,flush=True)
        with (candidate/(stage+'.log')).open('a',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((candidate/'promotion_comparison.json').read_text(encoding='utf-8'))
    write_json(candidate/'evaluation_progress.json',{'stage':'Evaluation complete; deployment review pending','metric_gate_passed':report['metric_gate_passed'],'deployed':False,'remaining':'English external regression comparison, independent confirmation, API/UI and deployment checks'})
    print('Evaluation complete. No automatic deployment.',flush=True)

if __name__=='__main__':main()
