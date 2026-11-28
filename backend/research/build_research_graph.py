"""Deterministic literature map from reviewed JSON; never invents citation edges."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
from research.extract_metadata import normalize_title
from research.validate_research_data import validate
ROOT=Path(__file__).resolve().parents[2]


def identity_keys(paper):
    keys={('title',normalize_title(paper['title']))}
    if paper.get('doi'): keys.add(('doi',paper['doi'].casefold().strip()))
    if paper.get('arxiv'): keys.add(('arxiv',re.sub(r'v\d+$','',paper['arxiv'])))
    return keys


def canonicalize(records):
    papers=[]; duplicates=[]; exclusions=[]
    for row in records:
        if re.search(r'\bmoje\b',row.get('title','')+' '+row.get('shortTitle',''),re.I):
            exclusions.append({'reason':'A source was excluded at the research owner’s request.'}); continue
        found=next((p for p in papers if identity_keys(p)&identity_keys(row)),None)
        if found:
            duplicates.append({'source':row['source'],'canonical_id':found['id']})
            for v in row.get('versions',[]):
                if v not in found['versions']:found['versions'].append(v)
        else:papers.append(dict(row))
    return sorted(papers,key=lambda p:p['id']),duplicates,exclusions


def build_relationships(papers):
    edges=[];by_title={normalize_title(p['title']):p for p in papers}
    for p in papers:
        edges.append({'id':'relevance-'+p['id'],'source':'hmeg','target':p['id'],'type':'relevance','reason':p['opportunity']})
        for r in p['references']:
            target=by_title.get(normalize_title(r['title']))
            if target and target['id']!=p['id']:
                identifier='citation-'+p['id']+'-'+target['id']
                if not any(e['id']==identifier for e in edges):
                    edges.append({'id':identifier,'source':p['id'],'target':target['id'],'type':'citation','reason':f"Explicit bibliographic reference in {p['shortTitle']}, PDF page {r['page']}.",'pages':[r['page']]})
    for a,b in combinations(papers,2):
        shared=sorted(set(a['tags'])&set(b['tags']))
        if len(shared)>=2:
            edges.append({'id':'similarity-'+a['id']+'-'+b['id'],'source':a['id'],'target':b['id'],'type':'similarity','reason':'Shared curated methodological tags: '+', '.join(shared)+'. See each paper’s methodology evidence. Qualitative connection, not a similarity percentage.'})
    return sorted(edges,key=lambda e:e['id'])


def read_json(path,default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def build(curation,output_root=ROOT):
    records=[]
    files=sorted(Path(curation).glob('*.json'),key=lambda p:(p.name!='base.json',p.name))
    for path in files:
        data=read_json(path,[])
        if isinstance(data,list):records.extend(data)
    papers,duplicates,exclusions=canonicalize(records);edges=build_relationships(papers);validate(papers,edges)
    inventories=[]
    for name in ['inventory.json','additional_downloads.json','additional_inventory.json']:
        data=read_json(output_root/'research_sources'/name,[])
        inventories.extend(data if isinstance(data,list) else data.get('files',[]))
    manifest={'schema_version':1,'scope':'English-language HMEG detection. External language capabilities remain outside scope.',
      'unique_papers':len(papers),'source_pdf_inventory_count':len(inventories),'readable_pdf_sources':sum(bool(r.get('pages')) for r in inventories),'damaged_originals_recovered':sum('recovered' in r.get('status','') for r in inventories),'missing_sources':read_json(output_root/'research_sources/source_gaps.json',[]),'duplicate_records_removed':len(duplicates),'duplicates':duplicates,'exclusions':exclusions,
      'categories':sorted(set(p['category'] for p in papers)),'relationships':dict(Counter(e['type'] for e in edges)),
      'component_connections':sum(len(p['components']) for p in papers),'methodology_diagrams':len(papers),
      'public_pdfs':sum(bool(p['publicPdf']) for p in papers),'abstracts_republished':sum(bool(p['abstract']) for p in papers),'original_figure_or_table_pages':sum(len(p['figures']) for p in papers),
      'graph_configuration':{'layout':'Deterministic d3-force initial coordinates; equal-size paper nodes','similarity':'At least two exact shared curated method tags','similarity_min_shared_tags':2,'similarity_percentages':False,'citations':'Explicit verified bibliographic reference plus source page only'},
      'source_notes':['Combined PDFs are inspected for individual paper boundaries; extraction only suggests locations for manual review.','Dates and claims refer to identified source versions. No citation counts are estimated.','Restricted or unverified-license PDFs are not publicly redistributed. The device-local reader does not upload files.']}
    directory=output_root/'backend/data/research';directory.mkdir(parents=True,exist_ok=True)
    artifacts={'papers.json':papers,'relationships.json':edges,'expert_connections.json':[{'paper_id':p['id'],'components':p['components'],'interpretation':p['opportunity']} for p in papers],
      'timeline.json':[{'paper_id':p['id'],'year':p['year'],'title':p['title']} for p in sorted(papers,key=lambda p:(p['year'] or 0,p['id']))],'research_manifest.json':manifest}
    for name,data in artifacts.items():(directory/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target=output_root/'frontend/src/data/research/library.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'papers':papers,'relationships':edges,'manifest':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--curation',type=Path,default=ROOT/'research_sources/curation');args=parser.parse_args()
    print(json.dumps(build(args.curation),ensure_ascii=False,indent=2))
