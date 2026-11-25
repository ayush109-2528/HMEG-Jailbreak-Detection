"""Train local fusion candidates on OOF training features, select on validation only."""
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from ml.ensemble.research_meta import ResearchMetaClassifier,KEYS
from scripts.tune_false_positives import metrics,predict
from scripts.train_public_corpus import write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
 folder=ROOT/'models/new/2f45a12a0c56ba7afe6f322c6d743511';out=folder/'fusion_retraining';out.mkdir(exist_ok=True)
 bundle=joblib.load(folder/'hmeg.joblib');csv=ROOT/'data/HMEG_275K_EnglishOnly_RealSource.csv'
 with csv.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==bundle['training']['dataset_sha256']
 df=pd.read_csv(csv,usecols=['prompt','label','research_split'],keep_default_na=False)
 train=df[df.research_split=='train'];val=df[df.research_split=='validation'];y=val.label.to_numpy();lengths=val.prompt.str.len().to_numpy()
 oof=np.load(folder/'oof_predictions.npz',allow_pickle=False)
 assert np.array_equal(oof['training_row_indices'],train.index.to_numpy())
 scores=np.column_stack([np.load(folder/('validation_'+k+'.npy')) for k in KEYS]);fast=scores[:,:2].mean(axis=1)
 baseline=json.loads((folder/'false_positive_tuning/validation_selection.json').read_text())['baseline']
 best=None;results=[]
 for depth,trees in ((3,160),(4,160),(5,240),(6,240)):
  for benign_weight in (1.,1.5,2.):
   name=f'd{depth}_n{trees}_b{benign_weight}'
   model=ResearchMetaClassifier('xgboost')
   model.model=XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.05,n_jobs=2,random_state=42,eval_metric='logloss',reg_lambda=5,min_child_weight=10,scale_pos_weight=1/benign_weight)
   model.fit(oof['scores'],train.label.to_numpy(),train.prompt.str.len().to_numpy())
   p=model.probabilities(scores,lengths);selected=None
   for low in (0,.05,.1,.2,.3):
    for high in (.7,.8,.9,.95,1.):
     routed=(fast>low)&(fast<high)
     for threshold in np.round(np.arange(.35,.811,.01),2):
      if not low<threshold<high:continue
      r={'name':name,'depth':depth,'trees':trees,'benign_weight':benign_weight,'low':low,'high':high,'final':float(threshold),'routing_rate':float(routed.mean()),**metrics(y,np.where(routed,p>=threshold,fast>=high))}
      if r['recall']<baseline['recall'] or r['f1']<baseline['f1']:continue
      if selected is None or (r['fpr'],-r['recall'])<(selected['fpr'],-selected['recall']):selected=r
   results.append({'name':name,'eligible_selection':selected})
   if selected and (best is None or (selected['fpr'],-selected['recall'])<(best['fpr'],-best['recall'])):
    best=selected;joblib.dump(model,out/'selected_meta.joblib',compress=3)
   write_json(out/'progress.json',{'completed':len(results),'total':12,'baseline':baseline,'best_validation':best,'deployed':False})
   print(json.dumps({'completed':len(results),'total':12,'selection':selected}),flush=True)
 selection={'baseline':baseline,'selected':best,'trials':results,'selection_scope':'OOF training fit; validation-only selection. No deployment.','deployed':False}
 write_json(out/'validation_selection.json',selection)
 if best and best['fpr']<baseline['fpr']:
  test=df[df.research_split=='test'];ts=np.column_stack([np.load(folder/('test_'+k+'.npy')) for k in KEYS]);tl=test.prompt.str.len().to_numpy();ty=test.label.to_numpy();route=bundle['routing_thresholds']
  previous,_=predict(ts,bundle['metas'][bundle['preferred_meta']],tl,route['low'],route['high'],route['final'])
  proposed,_=predict(ts,joblib.load(out/'selected_meta.joblib'),tl,best['low'],best['high'],best['final'])
  report={'baseline':metrics(ty,previous),'candidate':metrics(ty,proposed),'scope':'Previously inspected held-out regression set, cached batch predictions. Not fresh confirmation; actual serving and English OOD checks still required.','deployed':False}
  write_json(out/'test_regression.json',report);print(json.dumps(report),flush=True)
 print('Local fusion experiment complete. No live model changed.',flush=True)

if __name__=='__main__':main()
