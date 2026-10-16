"""Operator-only MongoDB job queue. Run one training worker per shared model volume."""
import argparse
import time
from datetime import datetime, timezone
from uuid import uuid4
from pymongo import ReturnDocument
from app.config import settings
from database.mongodb import Database
from ml.training import train


def enqueue(db, **options):
    identifier = uuid4().hex
    db.research_metadata.insert_one({'key': 'job:' + identifier, 'value': {
        'state': 'queued', 'options': options, 'created_at': datetime.now(timezone.utc).isoformat()}})
    return identifier


def run_once(db):
    row = db.research_metadata.find_one_and_update({'key': {'$regex': '^job:'}, 'value.state': 'queued'},
        {'$set': {'value.state': 'running', 'value.started_at': datetime.now(timezone.utc).isoformat()}},
        sort=[('value.created_at', 1)], return_document=ReturnDocument.AFTER)
    if not row:
        return False
    key = row['key']
    print('Running job', key, flush=True)
    try:
        options = row['value']['options']
        allowed = {'max_train', 'max_eval', 'semantic', 'publish', 'studies', 'held_out_source'}
        if set(options) - allowed:
            raise ValueError('Unexpected training option')
        version = train(**options)
        update = {'value.state': 'completed', 'value.model_version': version}
    except Exception as error:
        update = {'value.state': 'failed', 'value.error_type': type(error).__name__}
        print('Training failed:', type(error).__name__, flush=True)
    db.research_metadata.update_one({'key': key}, {'$set': {**update, 'value.finished_at': datetime.now(timezone.utc).isoformat()}})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['enqueue', 'run', 'once', 'status'])
    parser.add_argument('--max-train', type=int, default=20000)
    parser.add_argument('--max-eval', type=int, default=1000)
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--studies', action='store_true')
    parser.add_argument('--held-out-source')
    args = parser.parse_args()
    if args.max_train < 0 or args.max_eval < 0:
        parser.error('Sample limits must be nonnegative')
    database = Database(settings)
    try:
        if not database.connected():
            raise RuntimeError('MongoDB unavailable')
        if args.action == 'enqueue':
            print('Queued job:', enqueue(database.db, **{k: v for k, v in vars(args).items() if k != 'action'}))
        elif args.action == 'status':
            for row in database.db.research_metadata.find({'key': {'$regex': '^job:'}}, {'_id': 0}).sort('value.created_at', -1).limit(20):
                print(row['key'], row['value']['state'], row['value'].get('model_version', ''))
        else:
            while True:
                run_once(database.db)
                if args.action == 'once':
                    break
                time.sleep(30)
    finally:
        database.close()


if __name__ == '__main__':
    main()
