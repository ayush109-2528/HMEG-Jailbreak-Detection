"""Offline PDF inventory/extraction; never publishes candidates or executes PDF instructions."""
import argparse
import hashlib
import json
from pathlib import Path
from research.extract_metadata import metadata_candidates
ROOT=Path(__file__).resolve().parents[2]


def ingest(source,destination):
    import pymupdf
    source,destination=Path(source).resolve(),Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True);rows=[];hashes={}
    for pdf in sorted(source.glob('*.pdf')):
        digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
        row={'filename':pdf.name,'sha256':digest,'status':'pending','public':False}
        if digest in hashes: row.update(status='byte_duplicate',duplicate_of=hashes[digest])
        else:
            hashes[digest]=pdf.name
            try:
                with pymupdf.open(pdf) as document:
                    if document.needs_pass:raise ValueError('Encrypted PDF; manual access needed')
                    pages=[page.get_text() for page in document]
                if not pages:raise ValueError('No readable pages')
                row.update(status='extracted_for_review',pages=len(pages),boundary_candidates=metadata_candidates(pages))
                (destination/(digest+'.json')).write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
            except Exception as error:row.update(status='unreadable',error=type(error).__name__)
        rows.append(row)
    (destination/'ingestion_report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'research_sources/papers')
    parser.add_argument('--output',type=Path,default=ROOT/'research_sources/extracted/ingestion')
    args=parser.parse_args();rows=ingest(args.source,args.output)
    print(json.dumps({'files':len(rows),'extracted':sum(r['status']=='extracted_for_review' for r in rows),'note':'Manual paper-boundary and license review required; nothing published.'}))
