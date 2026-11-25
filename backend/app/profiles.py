"""Request-scoped research tracks. Credentials never leave the server."""
import os
from dataclasses import replace
from pathlib import Path
from threading import RLock
from fastapi import HTTPException
from app.config import ROOT
from database.mongodb import Database
from database.repositories import Repository
from ml.utils.model_info import ModelStore

SOURCE_VERSION='2f45a12a0c56ba7afe6f322c6d743511'
SOURCE_SHA='7f2914f9cf15d2345940facd75a56f18958c53f6732654a15aaa3d3f6337e50a'
CATALOG=[
 {'id':'synthetic','name':'Synthetic dataset','description':'Original augmented corpus and its archived measured results.','expected_samples':300000,'status':'Established research baseline'},
 {'id':'source','name':'Sourced dataset','description':'275K English prompts from recorded upstream sources. Includes upstream-generated data; not human-only.','expected_samples':275000,'status':'Research candidate · external false-positive gate not passed'}]

class SourceTrack:
 def __init__(self, settings):
  self.config=replace(settings,mongo_uri=os.getenv('SOURCE_MONGODB_URI') or os.getenv('MONGODB_MIGRATION_URI',''),database=os.getenv('SOURCE_MONGODB_DATABASE','hmeg_research'),model_dir=ROOT/os.getenv('SOURCE_MODEL_DIR','models/new/'+SOURCE_VERSION))
  self._database=None;self._repo=None;self._models=None;self.lock=RLock()
 @property
 def database(self):
  with self.lock:
   if self._database is None:self._database=Database(self.config)
   return self._database
 @property
 def repo(self):
  with self.lock:
   if self._repo is None:self._repo=Repository(self.database)
   return self._repo
 def model(self):
  with self.lock:
   if self._models is None:self._models=ModelStore(self.config)
   if self._models.bundle and self._models.bundle['version']!=SOURCE_VERSION:
    raise HTTPException(503,'Sourced track artifact does not match its recorded research version.')
   return self._models
 def repository(self):
  metadata=self.repo.metadata('dataset')
  if not metadata or metadata.get('sha256')!=SOURCE_SHA:
   raise HTTPException(503,'Sourced English dataset is not synchronized. Run the versioned English database preparation before deployment.')
  return self.repo
 def close(self):
  if self._database is not None:self._database.close()
