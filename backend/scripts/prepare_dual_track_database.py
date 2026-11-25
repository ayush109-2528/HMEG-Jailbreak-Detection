"""Prepare the sourced English demo database; preserve synthetic DB and Hindi archive."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from pymongo import ReplaceOne
from bson import BSON
from database.mongodb import Database
from scripts.stage_public_corpus import migration_settings,compact,cluster_bytes
from scripts.preflight_public_import import load_rows
from app.profiles import SOURCE_SHA,SOURCE_VERSION
ROOT=Path(__file__).resolve().parents[1]

def prepare():
 data=ROOT/'data';csv=data/'HMEG_275K_EnglishOnly_RealSource.csv';summary=json.loads((data/'HMEG_275K_EnglishOnly_RealSource_summary.json').read_text())
 with csv.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SOURCE_SHA
 originals=data/'HMEG_275K_English_Hindi_RealSource.csv'
 pairs=[];expected=set();sources=Counter();licenses=defaultdict(set);urls=defaultdict(set)
 for old,new in zip(load_rows(originals),load_rows(csv),strict=True):
  expected.add(new['normalized_sha256']);sources[new['source_dataset']]+=1;licenses[new['source_dataset']].add(new['license']);urls[new['source_dataset']].add(new['source_url'])
  if old['language']=='hi':pairs.append((old,new))
  else:assert old==new
 assert len(pairs)==616 and len(expected)==275000
 database=Database(migration_settings())
 try:
  if not database.connected():raise RuntimeError('Source database unavailable')
  db=database.db;metadata=db.research_metadata.find_one({'key':'dataset'})
  if not metadata or metadata['value'].get('sha256') not in (summary['original_sha256'],SOURCE_SHA):raise ValueError('Unexpected target dataset; no writes')
  if cluster_bytes(database.client)>512*1024**2-8*1024**2:raise ValueError('Insufficient safe archive headroom')
  old_docs={r['sid']:r for r in db.dataset_samples.find({'sid':{'$in':[o['sample_id'] for o,n in pairs]}})}
  if old_docs:
   db.dataset_hindi_archive.bulk_write([ReplaceOne({'sid':sid},row,upsert=True) for sid,row in old_docs.items()],ordered=True)
   assert db.dataset_hindi_archive.count_documents({})==616
   db.research_metadata.update_one({'key':'dataset'},{'$set':{'value.status':'updating_english'}})
   db.dataset_samples.bulk_write([ReplaceOne({'sid':old['sample_id'],'nh':old['normalized_sha256']},compact(new)) for old,new in pairs if old['sample_id'] in old_docs],ordered=True)
  assert db.dataset_samples.count_documents({})==275000
  assert db.dataset_samples.count_documents({'lang':'en'})==275000
  actual={r['nh'] for r in db.dataset_samples.find({},{'nh':1,'_id':0})}
  assert actual==expected,'Database hash membership mismatch'
  for old,new in pairs:
   got=db.dataset_samples.find_one({'sid':new['sample_id']},{'_id':0});assert got==compact(new)
  # Keep old metadata for reproducible rollback; the archived 616 rows restore v1.
  if metadata['value'].get('sha256')!=SOURCE_SHA:
   db.research_metadata.update_one({'key':'bilingual_dataset_archive'},{'$set':{'value':metadata['value']}},upsert=True)
  value={**summary,'sha256':SOURCE_SHA,'status':'ready','filename':csv.name,'source_counts':dict(sources),'dataset_version':'HMEG-275K-English-v2'}
  db.research_metadata.update_one({'key':'dataset'},{'$set':{'value':value}},upsert=True)
  db.research_metadata.update_one({'key':'data_quality'},{'$set':{'value':summary['integrity']}},upsert=True)
  for source,count in sources.items():
   db.dataset_sources.update_one({'dataset_name':source},{'$set':{'records':count,'dataset_version':value['dataset_version'],'licenses':sorted(licenses[source]),'source_urls':sorted(urls[source]),'provenance':'Pinned public-source records; upstream labels, synthetic/unknown authorship and license limitations retained.'}},upsert=True)
  folder=ROOT/'models/new'/SOURCE_VERSION;read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
  manifest=read('model_manifest.json');records=[]
  def add(name,kind,result,count=31690,**extra):
   identifier=hashlib.sha256((SOURCE_VERSION+kind+name).encode()).hexdigest()[:32]
   n=result.get('samples',count);timing=result.get('latency_samples',result.get('timing_samples', n if 'scope' in result and 'Actual warm' in result['scope'] else 0))
   records.append({**result,'experiment_id':identifier,'name':name,'model_type':kind,'model_version':SOURCE_VERSION,'created_at':manifest['created_at'],'dataset_version':manifest['dataset_version'],'dataset_sources':manifest['sources'],'training':manifest,'routing_thresholds':manifest['routing_thresholds'],'classification_sample_count':n,'latency_sample_count':timing,'evaluation_method':result.get('evaluation_mode',result.get('scope','Held-out cached batch predictions; latency absent unless measured separately.')),'bundle_size_bytes':(folder/'hmeg.joblib').stat().st_size,**extra})
  add('HMEG · sourced English test','hmeg',read('hmeg_results.json'))
  for name,r in read('expert_results.json').items():add(name.title()+' Expert','expert',r)
  for r in read('baseline_results.json'):add(r['name'],'baseline',r)
  for r in read('flat_results.json'):add(r['name'],'flat',r)
  for r in read('ablation_results.json'):add(r['name'],'ablation',r)
  for r in read('router_threshold_results.json'):add(f"Validation routing {r['low']} / {r['high']}",'routing',r,count=29357,routing_thresholds={k:r[k] for k in ('low','high','final')},evaluation_method='Validation selection experiment; not test performance.')
  for r in read('ood_results.json')['source_specific_results']:
   if r['source']=='CohereLabs/aya_redteaming':continue
   add(r['name']+' · '+r['source'],'ood',r,count=252 if 'NotInject' in r['source'] else 108,evaluation_method='Previously inspected English external challenge set. JBB positives are harmful-behavior proxies; exact overlaps excluded, semantic overlap remains possible.')
  assert sum(len(BSON.encode(row)) for row in records)<2*1024**2
  db.experiments.bulk_write([ReplaceOne({'experiment_id':r['experiment_id']},r,upsert=True) for r in records])
  result={'source_rows':275000,'english_rows':275000,'archived_hindi':616,'experiment_records':len(records),'source_names':len(sources),'synthetic_database_modified':False,'dataset_sha256':SOURCE_SHA,'status':'verified'}
  (data/'dual_track_database_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
 finally:database.close()

if __name__=='__main__':prepare()
