"""Validate a research CSV before resumable, bounded MongoDB upserts. No generated rows."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from bson import BSON
from pymongo import UpdateOne
from app.config import settings
from database.mongodb import Database
from database.collections import FIELDS
from database.indexes import create_indexes


def records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if not row.get('prompt', '').strip() or row.get('label') not in {'0', '1'}:
                raise ValueError(f'Invalid prompt/label at CSV line {line}')
            doc = {short: row[name] for name, short in FIELDS.items() if row.get(name, '') != ''}
            doc['l'] = int(row['label'])
            if row.get('class_name') and row['class_name'] != ('jailbreak' if doc['l'] else 'benign'):
                raise ValueError(f'Label/class disagreement at line {line}')
            if 'syn' in doc:
                if str(doc['syn']).lower() not in {'0', '1', 'true', 'false'}:
                    raise ValueError(f'Invalid synthetic flag at line {line}')
                doc['syn'] = str(doc['syn']).lower() in {'1', 'true'}
            doc.setdefault('ph', hashlib.sha256(doc['p'].encode()).hexdigest())
            doc.setdefault('nh', hashlib.sha256(' '.join(doc['p'].lower().split()).encode()).hexdigest())
            doc.setdefault('sid', row.get('record_id') or doc['ph'])
            doc.setdefault('src', row.get('source') or 'Unspecified')
            doc.setdefault('ac', row.get('category') or 'Unspecified')
            yield doc


def inspect(path):
    size, count, ids, hashes = 0, 0, set(), Counter()
    groups = {key: Counter() for key in ('l', 'src', 'syn', 'split', 'ver', 'ac')}
    for doc in records(path):
        if doc['sid'] in ids:
            raise ValueError('Duplicate sample ID; fix the CSV before importing.')
        ids.add(doc['sid'])
        hashes[doc['nh']] += 1
        count += 1
        size += len(BSON.encode(doc)) + 32
        for key in groups:
            groups[key][str(doc.get(key, 'unspecified'))] += 1
    if not count:
        raise ValueError('CSV is empty')
    digest = hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()
    return dict(filename=Path(path).name, sha256=digest, rows=count, estimated_document_bytes=size,
                estimated_with_index_margin_bytes=int(size * 1.4),
                duplicate_normalized_rows=sum(n - 1 for n in hashes.values()),
                distributions={key: dict(value) for key, value in groups.items()})


def import_csv(path, dry_run=False):
    report = inspect(path)
    print(json.dumps(report, indent=2), flush=True)
    if dry_run:
        return report
    database = Database(settings)
    try:
        if not database.connected():
            raise RuntimeError('MongoDB unavailable; check backend/.env and network access.')
        db = database.db
        prior = db.research_metadata.find_one({'key': 'dataset'})
        if db.dataset_samples.estimated_document_count() and (not prior or prior['value'].get('sha256') != report['sha256']):
            raise ValueError('Existing dataset differs. Use a separate database for a different dataset version.')
        if report['estimated_with_index_margin_bytes'] > 430 * 1024**2:
            raise ValueError('Estimated import exceeds conservative free-tier headroom; use a larger cluster or a smaller dataset.')
        create_indexes(db)
        metadata = {**report, 'status': 'importing', 'imported_rows': 0,
                    'timestamp': datetime.now(timezone.utc).isoformat()}
        db.research_metadata.update_one({'key': 'dataset'}, {'$set': {'value': metadata}}, upsert=True)
        batch, done = [], 0
        for doc in records(path):
            batch.append(UpdateOne({'sid': doc['sid']}, {'$setOnInsert': doc}, upsert=True))
            if len(batch) == 1000:
                db.dataset_samples.bulk_write(batch, ordered=False)
                done += len(batch)
                batch.clear()
                db.research_metadata.update_one({'key': 'dataset'}, {'$set': {'value.imported_rows': done}})
                if done % 10000 == 0:
                    print(f'Imported {done:,}/{report["rows"]:,}', flush=True)
        if batch:
            db.dataset_samples.bulk_write(batch, ordered=False)
            done += len(batch)
        if db.dataset_samples.count_documents({}) != report['rows']:
            raise RuntimeError('Post-import count mismatch; dataset remains marked importing.')
        for source, count in report['distributions']['src'].items():
            db.dataset_sources.update_one({'dataset_name': source}, {'$set': {
                'records': count, 'provenance': 'User-supplied CSV; source name is not independent verification.',
                'dataset_sha256': report['sha256']}}, upsert=True)
        db.research_metadata.update_one({'key': 'dataset'}, {'$set': {'value.status': 'ready', 'value.imported_rows': done}})
        db.research_metadata.update_one({'key': 'data_quality'}, {'$set': {'value': {
            'duplicate_normalized_rows': report['duplicate_normalized_rows'], 'invalid_rows': 0,
            'note': 'CSV labels and lineage retained as supplied. Synthetic records are not independent observations.'}}}, upsert=True)
        print(f'Verified {done:,} records in MongoDB.', flush=True)
        return report
    finally:
        database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        import_csv(args.csv, args.dry_run)
    except Exception as error:
        # Driver exceptions may contain infrastructure information; never print the URI.
        print(str(error) if isinstance(error, (ValueError, RuntimeError)) else f'Import stopped: {type(error).__name__}. Resume with the same CSV after resolving connectivity/quota issues.')
        raise SystemExit(1)
