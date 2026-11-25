import numpy as np
import pytest
from ml.oof import build_oof
from ml.ensemble.research_meta import meta_features, ResearchMetaClassifier
from ml.ensemble.flat_ensemble import fuse

def test_oof_never_shares_groups_and_predicts_once():
    labels=np.repeat([0,1]*10,2); groups=np.repeat(np.arange(20),2)
    calls=[]
    def predict(fold,train,test):
        assert not set(groups[train]) & set(groups[test])
        calls.append((fold,test.tolist()))
        return np.full((len(test),5),.25)
    scores,folds=build_oof(labels,groups,predict)
    assert len(calls)==5 and scores.shape==(40,5)
    assert sorted(i for _,rows in calls for i in rows)==list(range(40))
    assert all(len(set(folds[groups==g]))==1 for g in set(groups))

def test_oof_rejects_bad_scores():
    with pytest.raises(ValueError,match='probabilities'):
        build_oof([0,1]*10,range(20),lambda fold,train,test:np.full((len(test),5),np.nan))

def test_meta_features_and_required_context():
    scores=np.array([[.1,.3,.2,.4,.8],[.9,.8,.7,.6,.9]])
    features=meta_features(scores,[10,200])
    assert features.shape==(2,12)
    assert features[0,5]==pytest.approx(.2)
    assert features[0,-1]==10
    model=ResearchMetaClassifier().fit(scores,[0,1],[10,200])
    values=dict(zip(model.keys,scores[0]))
    assert 0<=fuse(values,'meta_logistic',model,prompt_length=10)<=1
    with pytest.raises(ValueError,match='prompt length'):
        fuse(values,'meta_logistic',model)
