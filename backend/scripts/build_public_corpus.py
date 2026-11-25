"""Build a local candidate only. Never mutates MongoDB or production models.
Pinned remote streams; bounded hash-priority sampling; source-held-out exclusions.
Run from backend: python -m scripts.build_public_corpus --max-scan 1000000
"""
import argparse
from collections import Counter
import csv
import heapq
import json
import re
import time
from pathlib import Path
from datasets import load_dataset
import fasttext
from scripts.corpus_integrity import describe, normalized, sha, split_for, validate

ROOT = Path(__file__).resolve().parents[1] / 'data'
REVISIONS = {'youbin2014/JailbreakDB':'2c2141bf74ebbe1fd3666a4b38be8856f6e47a92',
 'quickium/prompt-security-v0':'8d67314bd5c08497ed1f20a62009bec58e7c5eb0'}
# These sources remain outside training, including their copies in aggregations.
HELDOUT = ('notinject','harmbench','jbb','jailbreakbench','trustair','jailbreakhub','qualifire','in-the-wild')
SYNTHETIC = ('gcg','gpt','autodan','gbda','pair','pap','tap','uat','autoprompt','template',
 'wildjailbreak','rene','airoboros','camel','alpaca','openhermes','eclipse','adappa','h4rm3l')
PROXY = ('advbench','saferlhf','safe-rlhf','beavertails','hh-rlhf','techhazard','directrequest','maliciousinstruct')
QUICK_LICENSE = {'coconot':'odc-by-1.0', 'gandalf_ignore_instructions':'mit',
 'gandalf_summarization':'mit', 'yanismiraoui_prompt_injections':'apache-2.0',
 'deepset_prompt_injections':'apache-2.0 (card; dataset_info also lists cc-by-4.0)',
 'octavio_pi_multilingual':'gpl-3.0', 'gptfuzzer':'mit'}

class Builder:
 def __init__(self, max_scan):
  self.max_scan=max_scan; self.stats=Counter(); self.sources=Counter(); self.failures=[]
  self.seen={}; self.conflicts=set(); self.pool=[[],[]]; self.candidates={}
  path=ROOT/'lid.176.ftz'
  if not path.exists() or __import__('hashlib').sha256(path.read_bytes()).hexdigest()!='8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83':
   raise ValueError('Install the pinned official fastText lid.176.ftz asset; checksum mismatch or missing file.')
  self.detector=fasttext.load_model(str(path))
  external=ROOT/'NotInject_external.jsonl'
  self.external_hashes={json.loads(line)['normalized_sha256'] for line in external.read_text(encoding='utf-8').splitlines()} if external.exists() else set()
  self.started=time.time()

 def language(self, text):
  # Model supports 176 languages; do not constrain predictions to en/hi.
  values=self.detector.f.predict(' '.join(text.split())+'\n',1,0.0,'strict')
  if not values or values[0][0] < .70: return None
  language=values[0][1].removeprefix('__label__')
  if language=='en': return 'en'
  if language=='hi' and re.search('[\u0900-\u097f]',text): return 'hi'
  return None

 def add(self, row, repo, split, index):
  source=str(row.get('source') or 'unknown'); lower=source.lower()
  self.sources[repo+' / '+source]+=1
  if any(k in lower for k in HELDOUT): self.stats['held_out_source']+=1; return
  if 'spml' in lower or row.get('system_prompt'):
   self.stats['requires_system_context']+=1; return
  if str(row.get('domain','')).upper()=='PII': self.stats['pii_only']+=1; return
  prompt=row.get('user_prompt',row.get('text'))
  if not isinstance(prompt,str) or not prompt.strip(): self.stats['empty']+=1; return
  if len(prompt)>10000: self.stats['exceeds_serving_limit_10000_chars']+=1; return
  try: label=int(row.get('jailbreak',row.get('label')))
  except (ValueError,TypeError): self.stats['invalid_label']+=1; return
  if label not in (0,1): self.stats['invalid_label']+=1; return
  info=describe(prompt); digest=info['normalized_sha256']
  if digest in self.external_hashes: self.stats['external_exact_match']+=1; return
  if digest in self.seen:
   self.stats['duplicates_removed']+=1
   if self.seen[digest]!=label:
    self.conflicts.add(digest); self.candidates.pop(digest,None)
   return
  self.seen[digest]=label
  # Record language rejection; unknown must never be silently called English.
  language=self.language(prompt)
  if language is None: self.stats['language_or_confidence']+=1; return
  family='benign' if label==0 else ('harmful_behavior_proxy' if any(k in lower for k in PROXY) else 'unknown')
  if label and str(row.get('domain','')).upper()=='PI': family='prompt_injection'
  synthetic=True if any(k in lower for k in SYNTHETIC) else 'unknown'
  license=QUICK_LICENSE.get(source,'unknown') if 'quickium' in repo else 'cc-by-4.0 (aggregation; upstream terms not independently resolved)'
  if 'wildjailbreak' in lower: license='cc-by-nc-4.0 (upstream)'
  # Parent identifiers come only from source fields, never guessed from labels.
  parent=row.get('parent_id') or row.get('base_record_id') or row.get('group_id')
  group=source+':'+str(parent) if parent else 'text:'+digest
  record=dict(sample_id='hmeg275-'+digest[:24],prompt=prompt,label=label,
   class_name='security_positive' if label else 'benign', language=language,**info,
   attack_type=family,attack_family=family,source_category=str(row.get('tactic','unknown')),
   source_dataset=source, source_record_id=str(row.get('id') or 'unknown'),
   aggregation_source=repo,source_url='https://huggingface.co/datasets/'+repo,
   license=license,provenance=f'Upstream annotation; revision {REVISIONS[repo]}; split {split}; stream position {index}; not independently human-reviewed',
   is_synthetic=synthetic,group_id=group,source_revision=REVISIONS[repo])
  # Weighted hash priorities preserve scarce Hindi/short/keyword records.
  # This samples the declared scan window, not the entire multi-million corpus.
  weight=(4 if language=='hi' else 1)*(2 if info['is_short_prompt'] else 1)*(2 if info['special_keyword'] else 1)
  priority=int(sha('42:'+digest)[:16],16)/weight
  heap=self.pool[label]; cap=160000
  if len(heap)>=cap and priority>=-heap[0][0]: return
  if len(heap)>=cap:
   _,removed=heapq.heappop(heap); self.candidates.pop(removed,None)
  heapq.heappush(heap,(-priority,digest)); self.candidates[digest]=record
  self.stats['eligible_'+str(label)]+=1

 def stream(self,repo,split,config=None):
  scanned=0
  try:
   ds=load_dataset(repo,config,split=split,revision=REVISIONS[repo],streaming=True)
   for index,row in enumerate(ds):
    self.add(row,repo,split,index); scanned=index+1
    if scanned%25000==0:
     print(json.dumps({'source':repo,'split':split,'scanned':scanned,'candidates':len(self.candidates),'seconds':round(time.time()-self.started)}),flush=True)
    if scanned>=300000 and repo=='youbin2014/JailbreakDB' and len(self.pool[1 if split=='jailbreak' else 0])>=150000: break
    if scanned>=self.max_scan: break
  except Exception as exc:
   self.failures.append(dict(source=repo,split=split,error=type(exc).__name__,reason=str(exc)[:350]))
  return scanned

 def finish(self,scanned):
  # Conservative shared-prefix/suffix grouping keeps common prompt variants together.
  # Not a semantic-near-duplicate guarantee; record that limitation explicitly.
  rows=[r for k,r in self.candidates.items() if k not in self.conflicts]
  selected=[]
  for label in (0,1):
   items=[r for r in rows if r['label']==label]
   items.sort(key=lambda r: (r['language']!='hi', int(sha('selection42:'+r['normalized_sha256'])[:16],16)))
   selected.extend(items[:137500])
  parents=list(range(len(selected))); anchors={}
  def find(i):
   while parents[i]!=i: parents[i]=parents[parents[i]]; i=parents[i]
   return i
  for i,r in enumerate(selected):
   text=normalized(r['prompt']); keys=[r['group_id']]
   if len(text)>=200: keys += ['prefix:'+text[:160],'suffix:'+text[-160:]]
   for key in keys:
    if key in anchors: parents[find(i)]=find(anchors[key])
    else: anchors[key]=i
  group_names={}
  for i,r in enumerate(selected):
   root=find(i); group_names[root]=min(group_names.get(root,r['normalized_sha256']),r['normalized_sha256'])
  for i,r in enumerate(selected):
   r['group_id']=group_names[find(i)]; r['research_split']=split_for(r['group_id'])
  integrity=validate(selected)
  integrity['limitations']=['Unknown upstream lineage cannot be reconstructed. Prefix/suffix grouping is conservative and is not a complete semantic-near-duplicate audit.', 'Language classification is automatic, not manual verification.', 'JailbreakDB aggregate license does not resolve every upstream source license. License review is required before promotion.']
  # Record source-license uncertainty; no dataset bytes are published to Git.
  unresolved=sum('not independently resolved' in r['license'] or r['license']=='unknown' for r in selected)
  integrity['unresolved_upstream_license_rows']=unresolved
  # Aggregate terms are recorded separately; never call these individually audited upstream licenses.
  summary=dict(dataset_version='HMEG-275K-v1-candidate',status='validated' if integrity['passed'] else 'NOT_READY',
   total_rows=len(selected),positive_rows=sum(r['label'] for r in selected),benign_rows=sum(r['label']==0 for r in selected),
   language_detector={'name':'fastText lid.176.ftz','confidence_minimum':.70,'sha256':'8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83','license':'CC-BY-SA-3.0'},
   scanned=scanned, sampling='Seed 42 weighted hash-priority reservoir within bounded scan windows; Hindi retained first in final selection. Not a uniform sample of the full upstream corpus.',
   language_counts=dict(Counter(r['language'] for r in selected)),short_prompts=sum(r['is_short_prompt'] for r in selected),
   special_keyword_prompts=sum(r['special_keyword'] for r in selected),short_and_keyword=sum(r['special_keyword'] and r['is_short_prompt'] for r in selected),
   attack_family_counts=dict(Counter(r['attack_family'] for r in selected)),source_counts=dict(Counter(r['source_dataset'] for r in selected)),
   license_counts=dict(Counter(r['license'] for r in selected)),split_counts=dict(Counter(r['research_split'] for r in selected)),
   synthetic_counts=dict(Counter(str(r['is_synthetic']) for r in selected)),duplicates_removed=self.stats['duplicates_removed'],
   conflicting_labels_detected=len(self.conflicts),rejections=dict(self.stats),source_failures=self.failures,
   unresolved_license_rows=unresolved,elapsed_seconds=round(time.time()-self.started))
  ROOT.mkdir(parents=True,exist_ok=True)
  for name,value in [('summary',summary),('integrity_report',integrity)]:
   (ROOT/f'HMEG_275K_{name}.json').write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
  import pandas as pd
  frame=pd.DataFrame(selected)
  # Failed candidates are explicitly named, never the production dataset filename.
  stem='HMEG_275K_English_Hindi_RealSource' if integrity['passed'] else 'HMEG_275K_UNVERIFIED_candidate'
  if selected:
   frame.assign(matched_keywords=frame.matched_keywords.map(json.dumps),is_synthetic=frame.is_synthetic.map(str)).to_csv(ROOT/(stem+'.csv'),index=False)
   frame.assign(is_synthetic=frame.is_synthetic.map(str)).to_parquet(ROOT/(stem+'.parquet'),index=False)
   manifest=frame.groupby(['source_dataset','aggregation_source','source_revision','source_url','license'],dropna=False).size().reset_index(name='selected_rows')
   manifest.to_csv(ROOT/'HMEG_275K_source_manifest.csv',index=False)
  print(json.dumps({'status':summary['status'],'total':len(selected),'positive':summary['positive_rows'],'benign':summary['benign_rows'],'unresolved_licenses':unresolved,'failed_checks':[k for k,v in integrity['checks'].items() if not v]},indent=2),flush=True)
  return integrity['passed']

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--max-scan',type=int,default=1000000); args=parser.parse_args()
 builder=Builder(args.max_scan); scanned={}
 scanned['quickium/train']=builder.stream('quickium/prompt-security-v0','train')
 for split in ('jailbreak','regular'): scanned['JailbreakDB/'+split]=builder.stream('youbin2014/JailbreakDB',split,'prompt_corpus')
 raise SystemExit(0 if builder.finish(scanned) else 2)

if __name__=='__main__': main()
