"""Bounded public reads; writes are limited to consented inference results."""
import math
import re
import time
from threading import Lock
from uuid import uuid4
from datetime import datetime, timezone
from .collections import FIELDS, COLLECTIONS, readable


class Repository:
    def __init__(self, database):
        self.database = database
        self._cached = None
        self._expires = 0.0
        self._lock = Lock()

    @property
    def db(self):
        if self.database.db is None:
            raise ConnectionError('Database Offline')
        return self.database.db

    def metadata(self, key: str) -> dict | None:
        row = self.db.research_metadata.find_one({'key': key}, {'_id': 0})
        return row.get('value') if row else None

    def stats(self) -> dict:
        with self._lock:
            if self._cached is not None and time.monotonic() < self._expires:
                return self._cached
            facets = {key: [{'$group': {'_id': '$' + field, 'value': {'$sum': 1}}},
                            {'$sort': {'value': -1}}] for key, field in
                      [('labels', 'l'), ('sources', 'src'), ('categories', 'ac'), ('expert_focus', 'ef'),
                       ('splits', 'split'), ('synthetic', 'syn'), ('verification', 'ver')]}
            result = list(self.db.dataset_samples.aggregate([{'$facet': facets}], maxTimeMS=15000))[0]
            groups = {key: [{'name': item['_id'], 'value': item['value']} for item in rows]
                      for key, rows in result.items()}
            count = lambda field, name: next((x['value'] for x in groups[field] if x['name'] == name), 0)
            self._cached = dict(total=sum(x['value'] for x in groups['labels']), benign=count('labels', 0),
                                jailbreak=count('labels', 1), real=count('synthetic', False),
                                synthetic=count('synthetic', True), distributions=groups,
                                quality=self.metadata('data_quality'), dataset=self.metadata('dataset'))
            self._expires = time.monotonic() + 60
            return self._cached

    def samples(self, page: int, limit: int, search: str = '', **filters) -> dict:
        query = {FIELDS[key]: value for key, value in filters.items() if value is not None and value != ''}
        if search:
            query['p'] = {'$regex': re.escape(search), '$options': 'i'}
        total = self.db.dataset_samples.count_documents(query, maxTimeMS=12000)
        rows = self.db.dataset_samples.find(query, {'_id': 0}).sort('sid', 1).skip((page - 1) * limit).limit(limit).max_time_ms(12000)
        return dict(items=[readable(row) for row in rows], page=page, limit=limit,
                    total=total, pages=max(1, math.ceil(total / limit)))

    def sources(self) -> list:
        return list(self.db.dataset_sources.find({}, {'_id': 0}).sort('dataset_name', 1).limit(200))

    def experiments(self, model_type: str | None = None, search: str = '') -> list:
        query = {'model_type': model_type} if model_type else {}
        if search:
            query['name'] = {'$regex': re.escape(search), '$options': 'i'}
        return list(self.db.experiments.find(query, {'_id': 0}).sort('created_at', -1).limit(200))

    def experiment(self, experiment_id: str) -> dict | None:
        return self.db.experiments.find_one({'experiment_id': experiment_id}, {'_id': 0})

    def public_tests(self) -> list:
        return list(self.db.live_tests.find({'public_consent': True}, {'_id': 0}).sort('timestamp', -1).limit(20))

    def save_public_test(self, prompt: str, result: dict) -> str:
        identifier = uuid4().hex
        # The prompt is stored once; result contains no duplicate prompt field.
        self.db.live_tests.insert_one({'result_id': identifier, 'prompt': prompt, 'result': result,
            'public_consent': True, 'timestamp': datetime.now(timezone.utc).isoformat()})
        return identifier

    def public_test(self, identifier: str) -> dict | None:
        return self.db.live_tests.find_one({'result_id': identifier, 'public_consent': True}, {'_id': 0})

    def database_stats(self) -> dict:
        counts = {name: self.db[name].estimated_document_count() for name in COLLECTIONS}
        try:
            info = self.db.command('dbStats', scale=1)
            used = info['dataSize'] + info['indexSize']
            return dict(counts=counts, data_bytes=info.get('dataSize'), storage_bytes=info['storageSize'],
                        index_bytes=info['indexSize'], estimated_remaining_bytes=max(0, 512 * 1024**2 - used),
                        note='Approximate headroom using data + index size for this database only. Other databases share the cluster quota.')
        except Exception:
            return dict(counts=counts, storage_bytes=None, index_bytes=None, estimated_remaining_bytes=None,
                        note='Storage statistics unavailable with this database role. Counts are measured.')
