import numpy as np
from scripts.train_public_corpus import measures


def test_cached_score_metrics_do_not_fake_latency():
    result=measures(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
    assert result['metrics']['f1']==1.0
    assert result['metrics']['pr_auc']==1.0
    assert 'average_inference_time' not in result
    assert 'p95_inference_time' not in result


def test_benign_only_evaluation_does_not_claim_attack_recall_or_auc():
    result=measures(np.array([0,0]),np.array([.1,.2]))
    assert result['metrics']['recall'] is None
    assert result['metrics']['roc_auc'] is None
    assert result['metrics']['pr_auc'] is None
