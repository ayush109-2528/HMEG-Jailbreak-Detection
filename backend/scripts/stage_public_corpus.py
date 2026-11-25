"""Import only into a separately configured Atlas migration cluster.

No operation in this module deletes or writes to the current production cluster.
Staging is resumable by dataset checksum; activation refuses to replace an
existing collection. Application deployment remains a separate, later action.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from bson import BSON
from pymongo import UpdateOne
from app.config import settings
from database.collections import FIELDS
from database.mongodb import Database
from scripts.corpus_integrity import validate
from scripts.preflight_public_import import load_rows

STAGING = 'dataset_samples_new'
PRODUCTION = 'dataset_samples'
LIMIT = 512 * 1024**2
RESERVE = 64 * 1024**2


def migration_settings():
    uri = os.getenv('MONGODB_MIGRATION_URI', '')
    if not uri:
        raise ValueError('Set MONGODB_MIGRATION_URI in the ignored backend/.env.')
    target, current = urlsplit(uri).hostname, urlsplit(settings.mongo_uri).hostname
    if not target or target == current:
        raise ValueError('The migration URI must name a different Atlas cluster.')
    return replace(settings, mongo_uri=uri,
                   database=os.getenv('MONGODB_MIGRATION_DATABASE', 'hmeg_research'))


def cluster_bytes(client):
    total = 0
    for name in client.list_database_names():
        if name in ('admin', 'local', 'config'):
            continue
        stats = client[name].command('dbStats')
        total += stats['dataSize'] + stats['indexSize']
    return total


def compact(row):
    return {FIELDS.get(key, key): value for key, value in row.items()}


def verify(collection, rows):
    if collection.count_documents({}) != len(rows):
        raise ValueError('Staging count does not match the verified corpus.')
    for field in ('label', 'language', 'research_split', 'special_keyword', 'is_short_prompt'):
        expected = Counter(row[field] for row in rows)
        for value, count in expected.items():
            if collection.count_documents({FIELDS[field]: value}) != count:
                raise ValueError('Staging filter mismatch: ' + field)
    # Compare deterministic full documents at dispersed positions, not only the first batch.
    for index in range(0, len(rows), max(1, len(rows)//100)):
        expected = compact(rows[index])
        actual = collection.find_one({'sid': expected['sid']}, {'_id': 0})
        if actual != expected:
            raise ValueError('Staging document round-trip mismatch.')
    if collection.count_documents({'src': {'$in': ['', None]}}):
        raise ValueError('Missing source provenance.')



def complete_activation(db, path, rows, digest, report, *, rename):
    summary = json.loads((path.parent/'HMEG_275K_summary.json').read_text(encoding='utf-8'))
    metadata = {**summary, 'status': 'activating', 'sha256': digest, 'dataset_version': 'HMEG-275K-v1'}
    # BSON requires string mapping keys; validation class counts use integer labels.
    mongo_report = json.loads(json.dumps(report))
    db.research_metadata.update_one({'key': 'dataset'}, {'$set': {'value': metadata}}, upsert=True)
    db.research_metadata.update_one({'key': 'data_quality'}, {'$set': {'value': mongo_report}}, upsert=True)
    for source, count in summary['source_counts'].items():
        db.dataset_sources.update_one({'dataset_name': source}, {'$set': {'records': count,
            'dataset_version': 'HMEG-275K-v1',
            'provenance': 'Public-source corpus; upstream annotations and origin retained. Not independently human-reviewed.'}}, upsert=True)
    if rename:
        db[STAGING].rename(PRODUCTION, dropTarget=False)
    verify(db[PRODUCTION], rows)
    db.research_metadata.update_one({'key': 'dataset'}, {'$set': {'value.status': 'ready'}})
    db.research_metadata.update_one({'key': 'public_corpus_staging'}, {'$set': {'status': 'activated'}})
    result = {'status': 'activated_in_new_cluster', 'rows': len(rows), 'sha256': digest, 'old_cluster_modified': False}
    (path.parent/'HMEG_275K_import_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def stage(path, activate=False):
    path = Path(path)
    rows = list(load_rows(path))
    report = validate(rows)
    if not report['passed']:
        raise ValueError('Dataset integrity gate failed; no database writes performed.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    database = Database(migration_settings())
    try:
        if not database.connected():
            raise ValueError('Separate migration cluster is unreachable.')
        db = database.db
        state = db.research_metadata.find_one({'key': 'public_corpus_staging'})
        if PRODUCTION in db.list_collection_names():
            if not activate or not state or state.get('sha256') != digest or state.get('status') not in ('verified', 'activated'):
                raise ValueError('Target already has an unrelated production dataset; refusing replacement.')
            verify(db[PRODUCTION], rows)
            complete_activation(db, path, rows, digest, report, rename=False)
            return
        if state and state['sha256'] != digest:
            raise ValueError('Another dataset owns this staging collection.')
        if not state and STAGING in db.list_collection_names():
            raise ValueError('Unowned staging collection exists; refusing to modify it.')
        completed = state.get('completed', 0) if state else 0
        if not 0 <= completed <= len(rows):
            raise ValueError('Invalid staging checkpoint.')
        remaining = sum(len(BSON.encode(compact(row)))+17 for row in rows[completed:])
        used = cluster_bytes(database.client)
        # Verified staging already includes its indexes; activation adds only metadata.
        allowance = 2*1024**2 if state and state.get('status') == 'verified' and completed == len(rows) else RESERVE
        if used + remaining + allowance >= LIMIT:
            raise ValueError('Insufficient free-tier space for staging plus index/headroom allowance.')
        db.research_metadata.update_one({'key': 'public_corpus_staging'}, {'$setOnInsert':
            {'sha256': digest, 'completed': 0, 'status': 'importing'}}, upsert=True)
        collection = db[STAGING]
        collection.create_index('sid', unique=True)
        collection.create_index('nh', unique=True)
        for start in range(completed, len(rows), 1000):
            batch = rows[start:start+1000]
            collection.bulk_write([UpdateOne({'sid': row['sample_id']},
                {'$setOnInsert': compact(row)}, upsert=True) for row in batch], ordered=True)
            end = start + len(batch)
            db.research_metadata.update_one({'key': 'public_corpus_staging'}, {'$set': {'completed': end}})
            if end % 10000 == 0:
                print('Staged', end, '/', len(rows), flush=True)
                if cluster_bytes(database.client) >= LIMIT - 16*1024**2:
                    raise ValueError('Storage safety margin reached; staging preserved for inspection.')
        collection.create_index([('l', 1), ('lang', 1)])
        collection.create_index('split')
        collection.create_index('src')
        collection.create_index([('short', 1), ('kw', 1)])
        verify(collection, rows)
        db.research_metadata.update_one({'key': 'public_corpus_staging'}, {'$set': {'status': 'verified'}})
        if activate:
            complete_activation(db, path, rows, digest, report, rename=True)
            return
        result = {'status': 'activated_in_new_cluster' if activate else 'staging_verified',
                  'rows': len(rows), 'sha256': digest, 'old_cluster_modified': False}
        (path.parent/'HMEG_275K_import_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    finally:
        database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    try:
        stage(args.csv, args.activate)
    except Exception as exc:
        # Driver errors can contain connection details. Never echo them.
        print(str(exc) if isinstance(exc, ValueError) else
              'Staging failed (' + type(exc).__name__ + '); credentials suppressed.', flush=True)
        raise SystemExit(1)
