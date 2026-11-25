"""Local-only validation selection for lower FPR; never deploy or change live defaults."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scripts.train_public_corpus import write_json
from ml.ensemble.hmeg import KEYS

ROOT=Path(__file__).resolve().parents[1]

def metrics(y,pred):
    benign=y==0;positive=y==1
    fp=int(np.sum(pred & benign));tp=int(np.sum(pred & positive));fn=int(np.sum(~pred & positive))
    return {'fpr':fp/int(benign.sum()),'recall':tp/int(positive.sum()),'f1':2*tp/(2*tp+fp+fn),'fp':fp,'fn':fn,'benign':int(benign.sum()),'positive':int(positive.sum())}

def predict(scores,meta,lengths,low,high,threshold):
    fast=scores[:,:2].mean(axis=1);routed=(fast>low)&(fast<high)
    values=meta.probabilities(scores,lengths)
    labels=np.where(routed,values>=threshold,fast>=high)
    return labels,routed

def main():
    folder=ROOT/'models/new/2f45a12a0c56ba7afe6f322c6d743511'
    csv=ROOT/'data/HMEG_275K_EnglishOnly_RealSource.csv'
    bundle=joblib.load(folder/'hmeg.joblib')
    with csv.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==bundle['training']['dataset_sha256']
    data=pd.read_csv(csv,usecols=['prompt','label','research_split'],keep_default_na=False)
    val=data[data.research_split=='validation'];y=val.label.to_numpy();lengths=val.prompt.str.len().to_numpy()
    scores=np.column_stack([np.load(folder/('validation_'+k+'.npy')) for k in KEYS]);fast=scores[:,:2].mean(axis=1)
    route=bundle['routing_thresholds']
    pred,_=predict(scores,bundle['metas'][bundle['preferred_meta']],lengths,route['low'],route['high'],route['final']);baseline=metrics(y,pred)
    records=[]
    for name,meta in bundle['metas'].items():
        probabilities=meta.probabilities(scores,lengths)
        for low in (0,.025,.05,.1,.15,.2,.3):
            for high in (.7,.8,.9,.95,.975,1.):
                routed=(fast>low)&(fast<high)
                for threshold in np.round(np.arange(.30,.951,.01),2):
                    if not low<threshold<high:continue
                    labels=np.where(routed,probabilities>=threshold,fast>=high)
                    records.append({'meta':name,'low':low,'high':high,'final':float(threshold),'routing_rate':float(routed.mean()),**metrics(y,labels)})
    eligible=[r for r in records if r['recall']>=baseline['recall'] and r['f1']>=baseline['f1']]
    chosen=min(eligible,key=lambda r:(r['fpr'],-r['recall'],-r['f1'],r['routing_rate'])) if eligible else None
    out=folder/'false_positive_tuning';out.mkdir(exist_ok=True)
    selection={'selection_data':'Validation only','baseline':baseline,'selected':chosen,'trials':len(records),'constraint':'Recall and F1 must not fall below the current English candidate on validation; minimize FPR. Stronger than comparison with the weaker old model.','deployed':False,'target_fpr':.05,'limitations':'Previously examined test sets are regression checks, not a fresh independent confirmation. Frozen classifiers; this is routing/fusion selection, not new training.'}
    write_json(out/'validation_selection.json',selection)
    write_json(out/'validation_trials.json',records)
    # Freeze selection before loading any test predictions.
    if chosen and chosen['fpr']<baseline['fpr']:
        test=data[data.research_split=='test'];ty=test.label.to_numpy();tl=test.prompt.str.len().to_numpy()
        ts=np.column_stack([np.load(folder/('test_'+k+'.npy')) for k in KEYS])
        old,_=predict(ts,bundle['metas'][bundle['preferred_meta']],tl,route['low'],route['high'],route['final'])
        new,routed=predict(ts,bundle['metas'][chosen['meta']],tl,chosen['low'],chosen['high'],chosen['final'])
        report={'scope':'Previously inspected full held-out English regression set, using cached batched expert scores; not a new independent test or serving latency measurement.','baseline':metrics(ty,old),'selected':metrics(ty,new),'test_rows':len(test),'routing_rate':float(routed.mean()),'deployed':False}
        report['no_additional_misses']=report['selected']['fn']<=report['baseline']['fn']
        report['below_5_percent']=report['selected']['fpr']<.05
        write_json(out/'test_regression.json',report)
        print(json.dumps(report,indent=2))
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
