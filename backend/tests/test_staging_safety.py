from dataclasses import replace
import mongomock
import pytest
from scripts import stage_public_corpus as staging

def test_missing_or_same_cluster_is_rejected(monkeypatch):
    monkeypatch.delenv('MONGODB_MIGRATION_URI', raising=False)
    with pytest.raises(ValueError, match='Set MONGODB_MIGRATION_URI'):
        staging.migration_settings()
    monkeypatch.setattr(staging, 'settings', replace(staging.settings, mongo_uri='mongodb://localhost/old'))
    monkeypatch.setenv('MONGODB_MIGRATION_URI', 'mongodb://localhost/new')
    with pytest.raises(ValueError, match='different Atlas cluster'):
        staging.migration_settings()

def test_round_trip_checks_fields_and_filters():
    rows=[dict(sample_id='a',prompt='Example',label=0,language='en',research_split='test',special_keyword=False,is_short_prompt=True,source_dataset='public')]
    collection=mongomock.MongoClient().test.staging
    collection.insert_one(staging.compact(rows[0]))
    staging.verify(collection, rows)
    collection.update_one({}, {'$set': {'p':'changed'}})
    with pytest.raises(ValueError, match='round-trip mismatch'):
        staging.verify(collection, rows)


def test_activation_serializes_numeric_label_counts_and_marks_ready(tmp_path):
    import json
    db=mongomock.MongoClient().test
    rows=[dict(sample_id='a',prompt='Example',label=0,language='en',research_split='test',special_keyword=False,is_short_prompt=True,source_dataset='public')]
    db[staging.STAGING].insert_one(staging.compact(rows[0]))
    db.research_metadata.insert_one({'key':'public_corpus_staging','sha256':'fixture','status':'verified'})
    (tmp_path/'HMEG_275K_summary.json').write_text(json.dumps({'source_counts':{'public':1}}))
    report={'passed':True,'class_counts':{0:1,1:0}}
    staging.complete_activation(db,tmp_path/'fixture.csv',rows,'fixture',report,rename=True)
    assert db.research_metadata.find_one({'key':'data_quality'})['value']['class_counts']=={'0':1,'1':0}
    assert db.research_metadata.find_one({'key':'public_corpus_staging'})['status']=='activated'
    assert db.research_metadata.find_one({'key':'dataset'})['value']['status']=='ready'
    # Recovery is idempotent and never replaces the existing document.
    staging.complete_activation(db,tmp_path/'fixture.csv',rows,'fixture',report,rename=False)
    assert db[staging.PRODUCTION].count_documents({})==1
