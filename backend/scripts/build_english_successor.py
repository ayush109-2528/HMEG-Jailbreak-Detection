"""Create an English-only successor without modifying the original corpus or splits."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from scripts.build_public_corpus import Builder, REVISIONS
from scripts.preflight_public_import import load_rows
from scripts.corpus_integrity import normalized, sha, split_for, validate

ROOT=Path(__file__).resolve().parents[1]/'data'

def anchors(row):
    text=normalized(row['prompt'])
    return {'prefix:'+text[:160],'suffix:'+text[-160:]} if len(text)>=200 else set()

def main():
    original=ROOT/'HMEG_275K_English_Hindi_RealSource.csv'
    rows=list(load_rows(original)); retained=[r for r in rows if r['language']=='en']
    removed=[r for r in rows if r['language']=='hi']
    quotas=Counter((r['label'],r['research_split']) for r in removed)
    blocked={r['normalized_sha256'] for r in rows}
    for file in ('NotInject_external.jsonl','public_external_eval.jsonl'):
        blocked.update(json.loads(line)['normalized_sha256'] for line in (ROOT/file).read_text(encoding='utf-8').splitlines())
    occupied=set()
    for row in rows:occupied.update(anchors(row))
    builder=Builder(300000); added=[]; rejected=Counter(); scanned={}; used_parents=set()
    # Deterministic, pinned upstream order. Never generate or translate replacement text.
    for repo,config,split in [('quickium/prompt-security-v0',None,'train'),('youbin2014/JailbreakDB','prompt_corpus','regular'),('youbin2014/JailbreakDB','prompt_corpus','jailbreak')]:
        if not any(quotas.values()):break
        count=0
        for index,raw in enumerate(load_dataset(repo,config,split=split,revision=REVISIONS[repo],streaming=True)):
            count=index+1
            text=raw.get('user_prompt',raw.get('text'))
            if not isinstance(text,str):continue
            digest=sha(normalized(text))
            if digest in blocked:rejected['existing_or_external_hash']+=1;continue
            builder.add(raw,repo,split,index)
            row=builder.candidates.pop(digest,None)
            if row is None or row['language']!='en':continue
            parent=row['group_id']; keys=anchors(row)
            if keys & occupied or parent in used_parents:
                rejected['shared_structure_or_parent']+=1;continue
            row['group_id']=digest;row['research_split']=split_for(digest)
            key=(row['label'],row['research_split'])
            if quotas[key]<=0:continue
            row['sample_id']='hmeg275en-'+digest[:24]
            added.append(row);blocked.add(digest);occupied.update(keys);used_parents.add(parent);quotas[key]-=1
            if len(added)%100==0:print('Accepted replacements:',len(added),'/',len(removed),flush=True)
            if not any(quotas.values()):break
            if count>=300000:break
        scanned[repo+'/'+split]=count
    if any(quotas.values()):raise RuntimeError('Insufficient eligible replacements: '+str(quotas))
    # Replace in place, preserving all retained rows byte-for-byte at field level and their split assignments.
    buckets={key:iter([r for r in added if (r['label'],r['research_split'])==key]) for key in {(r['label'],r['research_split']) for r in removed}}
    output=[r if r['language']=='en' else next(buckets[(r['label'],r['research_split'])]) for r in rows]
    integrity=validate(output)
    integrity['checks'].update(english_only=all(r['language']=='en' for r in output),retained_rows_unchanged=all(a==b for a,b in zip(rows,output) if a['language']=='en'),split_label_counts_preserved=Counter((r['label'],r['research_split']) for r in rows)==Counter((r['label'],r['research_split']) for r in output))
    integrity['passed']=all(integrity['checks'].values())
    if not integrity['passed']:raise RuntimeError('Integrity check failed')
    stem='HMEG_275K_EnglishOnly_RealSource'
    target=ROOT/(stem+'.csv')
    if target.exists():raise FileExistsError('Refuse to overwrite existing successor corpus')
    def write_csv(path,records):
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            for row in records:writer.writerow({**row,'matched_keywords':json.dumps(row['matched_keywords'],ensure_ascii=False)})
    pending=target.with_suffix('.pending.csv');write_csv(pending,output);pending.replace(target)
    write_csv(ROOT/(stem+'_replacements.csv'),added)
    with original.open('rb') as stream:original_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    report={'dataset_version':'HMEG-275K-English-v2','status':'local_dataset_validated_not_trained_or_deployed','original_sha256':original_sha,'sha256':digest,'total_rows':len(output),'removed_hindi':len(removed),'added_english':len(added),'language_counts':dict(Counter(r['language'] for r in output)),'class_counts':dict(Counter(r['label'] for r in output)),'split_counts':dict(Counter(r['research_split'] for r in output)),'replacement_sources':dict(Counter(r['source_dataset'] for r in added)),'replacement_synthetic_flags':dict(Counter(str(r['is_synthetic']) for r in added)),'scanned':scanned,'rejections':dict(rejected),'integrity':integrity,'limitations':['Real source records means retrieved records, not verified human authorship. Upstream synthetic flags and license metadata retained.','Automatic language classification; no semantic near-duplicate guarantee.','English-only scope cannot establish Hindi performance. Existing external tests remain untouched.','Previous held-out results have been inspected; future model promotion needs an independent confirmation set.']}
    (ROOT/(stem+'_summary.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
