import csv
import mongomock
import pytest
from scripts import worker
from scripts.import_dataset import inspect, records


def test_worker_completes_claimed_job_once(monkeypatch):
    db=mongomock.MongoClient().hmeg_test
    calls=[]
    monkeypatch.setattr(worker,'train',lambda **kw: calls.append(kw) or 'test-version')
    identifier=worker.enqueue(db,max_train=20,max_eval=10,publish=False)
    assert worker.run_once(db) is True
    assert worker.run_once(db) is False
    record=db.research_metadata.find_one({'key':'job:'+identifier})
    assert record['value']['state']=='completed'
    assert record['value']['model_version']=='test-version'
    assert len(calls)==1


def test_worker_failure_is_recorded_without_exception_secrets(monkeypatch):
    db=mongomock.MongoClient().hmeg_test
    def fail(**kw): raise RuntimeError('private diagnostic')
    monkeypatch.setattr(worker,'train',fail)
    identifier=worker.enqueue(db,max_train=20)
    worker.run_once(db)
    value=db.research_metadata.find_one({'key':'job:'+identifier})['value']
    assert value['state']=='failed'
    assert value['error_type']=='RuntimeError'
    assert 'private diagnostic' not in str(value)


def test_import_preserves_provenance_and_rejects_duplicate_ids(tmp_path):
    p=tmp_path/'test.csv'
    columns=['sample_id','prompt','label','class_name','is_synthetic','source_dataset','base_record_id']
    with p.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        writer.writerow(['fixture-1','Explain clouds',0,'benign',0,'fixture','parent-1'])
    row=next(records(p))
    assert row['src']=='fixture' and row['bid']=='parent-1' and row['syn'] is False
    assert inspect(p)['rows']==1
    with p.open('a',newline='') as stream:
        csv.writer(stream).writerow(['fixture-1','Different prompt',1,'jailbreak',1,'fixture','parent-1'])
    with pytest.raises(ValueError,match='Duplicate sample ID'): inspect(p)
