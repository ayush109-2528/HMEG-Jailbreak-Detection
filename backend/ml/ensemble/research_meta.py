"""Versioned 12-feature fusion for the new OOF experiment; old bundles unchanged."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

KEYS=('word','format','meaning','pattern','instruction')
FEATURES=(*KEYS,'fast_score','expert_mean','expert_max','expert_min','expert_variance','expert_disagreement','prompt_length')

def meta_features(probabilities,lengths):
    p=np.asarray(probabilities,dtype=float); lengths=np.asarray(lengths,dtype=float)
    if p.ndim!=2 or p.shape[1]!=5 or lengths.shape!=(len(p),):
        raise ValueError('Expected Nx5 scores and N prompt lengths.')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.isfinite(lengths).all() or (lengths<0).any():
        raise ValueError('Invalid meta-feature inputs.')
    votes=(p>=.5).mean(axis=1)
    return np.column_stack([p,(p[:,0]+p[:,1])/2,p.mean(axis=1),p.max(axis=1),p.min(axis=1),p.var(axis=1),2*np.minimum(votes,1-votes),lengths])

class ResearchMetaClassifier:
    requires_prompt_length=True
    def __init__(self,kind='logistic'):
        if kind not in ('logistic','xgboost'): raise ValueError('Unknown meta classifier.')
        self.kind=kind; self.keys=list(KEYS); self.feature_names=list(FEATURES)
        self.model=(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)) if kind=='logistic' else XGBClassifier(n_estimators=80,max_depth=3,learning_rate=.08,n_jobs=1,random_state=42,eval_metric='logloss'))
    def fit(self,oof_scores,labels,lengths):
        self.model.fit(meta_features(oof_scores,lengths),labels)
        return self
    def probabilities(self,scores,lengths):
        return self.model.predict_proba(meta_features(scores,lengths))[:,list(self.model.classes_).index(1)]
    def score(self,scores,*,prompt_length):
        return float(self.probabilities([[scores[k] for k in KEYS]],[prompt_length])[0])
