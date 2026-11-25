import mongomock
from types import SimpleNamespace
from fastapi.testclient import TestClient
from app import main
from app.profiles import SOURCE_SHA,SOURCE_VERSION
from tests.test_core import StubExpert,StubMeta
from ml.ensemble.hmeg import HMEG,KEYS


def test_tracks_are_request_scoped_and_invalid_selection_fails(monkeypatch,tmp_path):
 class DB:
  def __init__(self,config):self.db=mongomock.MongoClient().research
  def connected(self):return True
  def close(self):pass
 monkeypatch.setattr(main,'Database',DB)
 from ml.utils.model_info import ModelStore
 from app.config import Settings
 monkeypatch.setattr(main,'ModelStore',lambda config:ModelStore(Settings(model_dir=tmp_path)))
 monkeypatch.setattr(main,'artifact_mtime',lambda:None)
 with TestClient(main.app) as client:
  main.visits.clear()
  old=client.app.state.database.db;old.dataset_samples.insert_one({'sid':'old','p':'old prompt','l':0})
  track=client.app.state.source_track;track._database=DB(None)
  track.database.db.dataset_samples.insert_one({'sid':'new','p':'new prompt','l':1})
  track.database.db.research_metadata.insert_one({'key':'dataset','value':{'sha256':SOURCE_SHA}})
  oldengine=HMEG({k:StubExpert(.1) for k in KEYS},StubMeta(KEYS),version='old')
  newengine=HMEG({k:StubExpert(.9) for k in KEYS},StubMeta(KEYS),version=SOURCE_VERSION)
  client.app.state.models.engine=oldengine
  track._models=SimpleNamespace(engine=newengine,bundle={'version':SOURCE_VERSION},status=lambda:{'models_loaded':True,'model_version':SOURCE_VERSION})
  assert client.get('/api/dataset/samples?profile=synthetic').json()['items'][0]['prompt']=='old prompt'
  assert client.get('/api/dataset/samples?profile=source').json()['items'][0]['prompt']=='new prompt'
  for profile,label in [('source',1),('synthetic',0),('source',1),('synthetic',0)]:
   r=client.post('/api/predict/hmeg?profile='+profile,json={'prompt':'Please explain the instructions in this document.'}).json()
   assert r['label']==label and r['research_profile']==profile and not r['saved']
  assert client.get('/api/health?profile=unknown').status_code==400
  track.database.db.research_metadata.update_one({'key':'dataset'},{'$set':{'value.sha256':'wrong'}})
  assert client.get('/api/dataset/samples?profile=source').status_code==503
  assert client.get('/api/dataset/samples?profile=synthetic').status_code==200
