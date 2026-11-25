"""Leakage-checked OOF predictions for training-only stacking.

The caller provides a fresh fit/predict operation per fold. It can reuse frozen
sentence embeddings, but must fit each classifier and learned vectorizer on the
fold training indices only. Validation and final test rows are never accepted
as extra inputs to this operation.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def build_oof(labels, groups, fit_predict, *, folds=5, seed=42):
    labels=np.asarray(labels); groups=np.asarray(groups)
    if len(labels)!=len(groups) or set(labels.tolist())!={0,1}:
        raise ValueError('OOF requires equally sized binary labels and group IDs.')
    if len(set(groups))<folds:
        raise ValueError('Not enough independent groups for the requested folds.')
    result=np.full((len(labels),5),np.nan); assigned=np.zeros(len(labels),dtype=int)
    fold_ids=np.full(len(labels),-1,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    for fold,(train,heldout) in enumerate(splitter.split(np.zeros(len(labels)),labels,groups)):
        if set(groups[train]) & set(groups[heldout]):
            raise ValueError('OOF group leakage detected.')
        if set(labels[train])!={0,1} or set(labels[heldout])!={0,1}:
            raise ValueError('Each OOF fold must contain both classes.')
        scores=np.asarray(fit_predict(fold,train,heldout),dtype=float)
        if scores.shape!=(len(heldout),5) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
            raise ValueError('Expected five finite expert probabilities per held-out row.')
        result[heldout]=scores; assigned[heldout]+=1; fold_ids[heldout]=fold
    if not np.all(assigned==1) or not np.isfinite(result).all():
        raise ValueError('Every training row must receive exactly one OOF prediction.')
    return result,fold_ids
