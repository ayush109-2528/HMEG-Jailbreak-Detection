"""Read-only capacity gate for an Atlas staging import. Never drops collections."""
import argparse
import csv
import json
from pathlib import Path
from bson import BSON
from app.config import settings
from database.mongodb import Database
from scripts.corpus_integrity import validate

from database.collections import FIELDS


def load_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            row['label']=int(row['label'])
            for key in ('word_count','char_count'):row[key]=int(row[key])
            for key in ('is_short_prompt','special_keyword'):row[key]=row[key].lower()=='true'
            row['matched_keywords']=json.loads(row['matched_keywords'])
            if row['is_synthetic'].lower() in ('true','false'):row['is_synthetic']=row['is_synthetic'].lower()=='true'
            yield row

def preflight(path):
    rows=list(load_rows(path)); integrity=validate(rows)
    if not integrity['passed']:return {'status':'blocked_integrity','integrity':integrity}
    size=sum(len(BSON.encode({FIELDS.get(k,k):v for k,v in row.items()}))+17 for row in rows)
    database=Database(settings)
    try:
        if not database.connected():return {'status':'blocked_connection','candidate_bson_bytes':size}
        # Atlas free quota counts logical BSON plus index bytes, not compressed storageSize.
        current=0
        for name in database.client.list_database_names():
            if name in ('admin','local','config'):continue
            stats=database.client[name].command('dbStats')
            current+=stats['dataSize']+stats['indexSize']
        reserve=64*1024*1024
        total=current+size+reserve
        return {'status':'capacity_available' if total<512*1024*1024 else 'blocked_capacity',
                'current_data_and_indexes_bytes':current,'candidate_bson_bytes':size,
                'estimated_new_index_and_headroom_bytes':reserve,'projected_total_bytes':total,
                'free_tier_limit_bytes':512*1024*1024,'rows':len(rows),
                'note':'Read-only estimate. Existing production data must remain until staging and API verification complete.'}
    finally:database.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');args=parser.parse_args()
    result=preflight(args.csv)
    target=Path(__file__).resolve().parents[1]/'data'/'HMEG_275K_mongodb_preflight.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
