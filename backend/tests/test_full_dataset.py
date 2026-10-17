import pandas as pd
import pytest
from ml.ensemble.hmeg import HMEG
from ml.evaluation.full_dataset import score_test_set, evaluate_full
from ml.training import evaluate

class Expert:
    def __init__(self, offset): self.offset=offset
    def score(self, text): return {'benign': .1, 'uncertain': .5, 'attack': .9}[text] + self.offset
    def inspect(self, text): return {}
class Meta:
    def score(self, scores): return sum(scores.values()) / len(scores)

@pytest.mark.parametrize('flat', [False, True])
def test_cached_full_test_matches_real_predictions_and_samples_only_latency(flat):
    experts={key:Expert(0) for key in ['word','format','meaning','pattern','instruction']}
    engine=HMEG(experts,Meta())
    frame=pd.DataFrame({'p':['benign','uncertain','attack'], 'l':[0,1,1]})
    cache=score_test_set(experts,frame)
    direct=evaluate(engine,frame,flat=flat)
    full=evaluate_full(engine,frame,cache,2,flat=flat)
    assert full['metrics']==direct['metrics']
    assert full['confusion_matrix']==direct['confusion_matrix']
    assert full['average_experts_executed']==direct['average_experts_executed']
    assert full['layer2_routing_rate']==direct['layer2_routing_rate']
    assert full['classification_sample_count']==3
    assert full['latency_sample_count']==2


def test_zero_limits_keep_every_eligible_row_and_remove_cross_split_leakage():
    from types import SimpleNamespace
    from ml.training import load_data
    rows=[]
    for split in ['train','validation','test']:
        for i in range(6):
            key=f'{split}-{i}'
            rows.append({'p':key,'l':i%2,'split':split,'src':'unit-fixture',
                         'nh':key,'bid':key,'pid':'','sid':key,'syn':False})
    rows[0]['nh']='test-0'
    rows[6]['bid']='test-0'
    db=SimpleNamespace(dataset_samples=SimpleNamespace(find=lambda *args: rows))
    parts, manifest=load_data(db,0,0)
    assert manifest['original_split_counts']=={'train':6,'validation':6,'test':6}
    assert manifest['used_split_counts']=={'train':5,'validation':5,'test':6}
    for left,right in [(0,1),(0,2),(1,2)]:
        assert not set(parts[left].nh) & set(parts[right].nh)
        assert not set(parts[left].group) & set(parts[right].group)
