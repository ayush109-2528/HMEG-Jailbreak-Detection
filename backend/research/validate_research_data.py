"""Validate curated literature evidence before generating public artifacts."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = {'classical','semantic','ensemble','structural','multilingual','evaluation','llm','attacks'}
COMPONENTS = {'word','format','meaning','pattern','instruction','router','meta','evaluation'}
REQUIRED = set('id title shortTitle authors year venue type doi arxiv url pdfUrl publicPdf source pages category tags components languages abstract abstractPermission summary objective methodology models features datasets results metrics limitations opportunity findings scopeNote direct indirect ensemble calibration routing steps evidence references license licenseUrl versions figures'.split())


def safe_url(value, local=False):
    if not isinstance(value,str) or '\\' in value or any(ord(c)<32 for c in value): return False
    if local and value.startswith('/research/'):
        return '..' not in unquote(value).split('/') and bool(re.fullmatch(r'/research/[a-zA-Z0-9_./-]+',value))
    p=urlparse(value)
    return p.scheme=='https' and bool(p.hostname) and not p.username and not p.password


def permits_publication(paper):
    return bool(re.fullmatch(r'CC BY(?:-SA)? (?:3\.0|4\.0)|CC0(?: 1\.0)?|PUBLIC DOMAIN',paper.get('license','').upper()))


def validate(papers,relationships=None,check_assets=True):
    errors=[]; ids=set()
    for p in papers:
        identifier=p.get('id','<missing>')
        def fail(message): errors.append(f'{identifier}: {message}')
        missing=REQUIRED-set(p)
        if missing: fail('missing fields: '+', '.join(sorted(missing))); continue
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*',identifier): fail('unsafe id')
        if identifier in ids: fail('duplicate id')
        ids.add(identifier)
        if re.search(r'\bmoje\b',p['title']+' '+p['shortTitle'],re.I): fail('excluded paper')
        if p['category'] not in CATEGORIES: fail('unknown category')
        if set(p['components'])-COMPONENTS: fail('unknown component')
        if not isinstance(p['pages'],int) or p['pages']<1: fail('invalid PDF page count'); continue
        if not p['authors'] or not all(isinstance(a,str) and a.strip() for a in p['authors']): fail('verified authors required')
        if p['year'] is not None and (not isinstance(p['year'],int) or not 1900<=p['year']<=2100): fail('invalid year')
        if len(p['summary'].split())<60: fail('summary too short')
        for key in ['url','pdfUrl','licenseUrl']:
            if p[key] and not safe_url(p[key]): fail('unsafe '+key)
        if p['doi'] and not re.fullmatch(r'10\.\d{4,9}/\S+',p['doi']): fail('invalid DOI')
        if p['arxiv'] and not re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?',p['arxiv']): fail('invalid arXiv identifier')
        for v in p['versions']:
            if not safe_url(v.get('url')): fail('unsafe version URL')
        for field,pages in p['evidence'].items():
            if not pages or any(not isinstance(n,int) or not 1<=n<=p['pages'] for n in pages): fail('invalid evidence pages: '+field)
        for field in ['metadata','methodology','results','limitations']:
            if field not in p['evidence'] and p.get(field) != 'Not specified in the source.': fail('missing page evidence: '+field)
        if p['abstract'] and (not permits_publication(p) or not p['abstractPermission']): fail('abstract permission or attribution missing')
        if p['publicPdf']:
            if not permits_publication(p): fail('public PDF permission not established')
            if not safe_url(p['publicPdf'],True) or not p['publicPdf'].startswith('/research/papers/'): fail('unsafe public PDF path')
            elif check_assets and not (ROOT/'frontend/public'/p['publicPdf'].lstrip('/')).is_file(): fail('missing public PDF asset')
        for f in p['figures']:
            if not 1<=f.get('page',0)<=p['pages']: fail('invalid figure page')
            if f.get('image'):
                if not permits_publication(p): fail('figure permission not established')
                if not safe_url(f['image'],True): fail('unsafe figure URL')
                elif check_assets and not (ROOT/'frontend/public'/f['image'].lstrip('/')).is_file(): fail('missing figure asset')
        for r in p['references']:
            if not r.get('title') or not 1<=r.get('page',0)<=p['pages']: fail('invalid reference evidence')
            if r.get('url') and not safe_url(r['url']): fail('unsafe reference URL')
    for e in relationships or []:
        if e['source'] not in ids|{'hmeg'} or e['target'] not in ids|{'hmeg'}: errors.append('Unknown relationship endpoint')
        if e['type'] not in {'citation','similarity','relevance'}: errors.append('Unknown relationship type')
        if not e.get('reason'): errors.append('Missing relationship rationale')
        if e['type']=='citation' and not e.get('pages'): errors.append('Citation lacks verified source page')
    if errors: raise ValueError('\n'.join(errors))
    return {'valid':True,'papers':len(papers),'relationships':len(relationships or [])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',nargs='?',type=Path,default=ROOT/'frontend/src/data/research/library.json')
    args=parser.parse_args(); data=json.loads(args.library.read_text(encoding='utf-8'))
    print(json.dumps(validate(data['papers'],data.get('relationships',[]))))
