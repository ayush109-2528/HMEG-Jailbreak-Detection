"""Researcher/worker training only; separate partitions for experts, fusion, and final evaluation."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import settings
from database.mongodb import Database
from ml.experts import CLASSES
from ml.ensemble.meta_classifier import MetaClassifier
from ml.ensemble.hmeg import HMEG
from ml.ensemble.router import Router
from ml.evaluation.metrics import summarize
from ml.evaluation.ood import source_holdout
from ml.evaluation.ablation import without_each
from ml.evaluation.routing_analysis import routers


def load_data(db, max_train=20000, max_eval=1000, held_out_source=None):
    frame = pd.DataFrame(list(db.dataset_samples.find({}, {'_id': 0, 'p': 1, 'l': 1, 'split': 1, 'src': 1, 'nh': 1, 'bid': 1, 'pid': 1, 'sid': 1, 'syn': 1})))
    if frame.empty:
        raise ValueError('Import a dataset before training.')
    frame = frame.fillna('')
    frame['group'] = frame.apply(lambda r: r.get('bid') or r.get('pid') or r.get('nh') or r['sid'], axis=1)
    if held_out_source:
        remaining, test = source_holdout(frame, held_out_source)
        # Use recorded validation roles; never tune on the held-out source.
        train, validation = remaining[remaining.split == 'train'], remaining[remaining.split == 'validation']
    else:
        train, validation, test = (frame[frame.split == value].copy() for value in ('train', 'validation', 'test'))
    original = {'train': len(train), 'validation': len(validation), 'test': len(test)}
    # Prioritize the held-out test partition when removing shared lineage / normalized text.
    validation = validation[~validation.group.isin(test.group) & ~validation.nh.isin(test.nh)]
    train = train[~train.group.isin(pd.concat([validation.group, test.group])) & ~train.nh.isin(pd.concat([validation.nh, test.nh]))]
    parts = []
    for part, cap in [(train, max_train), (validation, max_eval), (test, max_eval)]:
        part = part.drop_duplicates('nh')
        if set(part.l) != {0, 1} or part.l.value_counts().min() < 2:
            raise ValueError('Each disjoint partition needs at least two samples per class.')
        if cap and len(part) > cap:
            part, _ = train_test_split(part, train_size=cap, stratify=part.l, random_state=42)
        parts.append(part)
    return parts, {'original_split_counts': original, 'used_split_counts': dict(zip(['train', 'validation', 'test'], map(len, parts))),
        'sampling_seed': 42, 'max_train': max_train, 'max_eval': max_eval, 'held_out_source': held_out_source,
        'sources': sorted(frame.src.unique().tolist()), 'leakage_policy': 'Remove shared parent groups and normalized hashes across splits; deduplicate normalized text before sampling.'}


def evaluate(engine, test, *, flat=False, fusion='meta_xgboost'):
    results = [engine.predict(text, flat=flat, fusion=fusion) for text in test.p]
    out = summarize(test.l.tolist(), [r['label'] for r in results], [r['final_probability'] for r in results], [r['total_latency_ms'] for r in results])
    out.update(layer1_exit_rate=float(np.mean([r['decision_layer'] == 1 for r in results])),
        layer2_routing_rate=float(np.mean([r['routing']['routed_to_layer_2'] for r in results])),
        average_experts_executed=float(np.mean([r['experts_executed'] for r in results])),
        average_layer1_time=float(np.mean([r['layer1_time_ms'] for r in results])),
        average_layer2_time=float(np.mean([r['layer2_time_ms'] for r in results])))
    return out


def train(*, max_train=20000, max_eval=1000, semantic=False, publish=False, held_out_source=None, studies=False):
    started = time.perf_counter()
    database = Database(settings)
    try:
        if not database.connected():
            raise ValueError('MongoDB unavailable.')
        db = database.db
        meta = db.research_metadata.find_one({'key': 'dataset'})
        if not meta or meta['value'].get('status') != 'ready':
            raise ValueError('Dataset import must finish before training.')
        (train_set, val, test), manifest = load_data(db, max_train, max_eval, held_out_source)
        manifest['dataset_sha256'] = meta['value']['sha256']
        experts = {}
        for key, cls in CLASSES.items():
            if key == 'meaning' and not semantic:
                continue
            print(f'Training {key} expert on {len(train_set):,} rows', flush=True)
            experts[key] = cls().fit(train_set.p, train_set.l)
        print('Fitting learned fusion on validation partition', flush=True)
        columns = {key: expert.probabilities(val.p) for key, expert in experts.items()}
        rows = [{key: float(value[i]) for key, value in columns.items()} for i in range(len(val))]
        metas = {kind: MetaClassifier(list(experts), kind).fit(rows, val.l.tolist()) for kind in ['logistic', 'xgboost']}
        version = uuid4().hex
        manifest.update(available_experts=list(experts), reduced_expert_set=len(experts) < 5,
            training_seconds=time.perf_counter() - started, created_at=datetime.now(timezone.utc).isoformat())
        bundle = dict(experts=experts, metas=metas, preferred_meta='xgboost', version=version, training=manifest)
        engine = HMEG(experts, metas['xgboost'], Router(settings.low, settings.high), settings.threshold, version)
        results = []

        def save(name, kind, measurement, extra=None):
            row = dict(experiment_id=uuid4().hex, name=name, model_type='ood' if held_out_source else kind,
                model_version=version, created_at=manifest['created_at'], dataset_sources=manifest['sources'],
                features={key: expert.method for key, expert in experts.items()}, training=manifest,
                routing_thresholds={'low': settings.low, 'high': settings.high, 'final': settings.threshold},
                hyperparameters={'seed': 42, 'meta': 'xgboost', 'expert_parameters': {key: str(expert.model.get_params(deep=False)) for key, expert in experts.items()}},
                training_time=manifest['training_seconds'], **measurement)
            row.update(extra or {})
            db.experiments.insert_one(dict(row))
            results.append(row)

        save('HMEG' + (' (four-expert configuration)' if len(experts) < 5 else ''), 'hmeg', evaluate(engine, test))
        for key, expert in experts.items():
            timings, probabilities = [], []
            for text in test.p:
                before = time.perf_counter()
                probabilities.append(expert.score(text))
                timings.append((time.perf_counter() - before) * 1000)
            save(key.title() + ' Expert', 'expert', summarize(test.l.tolist(), [int(x >= .5) for x in probabilities], probabilities, timings))
        for method in ['majority', 'average', 'weighted', 'meta_logistic', 'meta_xgboost']:
            candidate = HMEG(experts, metas['logistic' if method == 'meta_logistic' else 'xgboost'], version=version)
            save('Flat ' + method.replace('_', ' ').title(), 'flat', evaluate(candidate, test, flat=True, fusion=method))
        if studies:
            for key, subset in without_each(experts):
                meta_model = MetaClassifier(list(subset)).fit([{k: r[k] for k in subset} for r in rows], val.l.tolist())
                save('HMEG without ' + key.title(), 'ablation', evaluate(HMEG(subset, meta_model), test))
            for router in routers():
                low, high = router.low, router.high
                save(f'Routing {low} / {high}', 'routing', evaluate(HMEG(experts, metas['xgboost'], Router(low, high)), test), {'routing_thresholds': {'low': low, 'high': high, 'final': .5}})
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        candidate = settings.model_dir / f'{version}.joblib'
        joblib.dump(bundle, candidate, compress=3)
        artifact_size = candidate.stat().st_size
        for row in results:
            db.experiments.update_one({'experiment_id': row['experiment_id']}, {'$set': {'bundle_size_bytes': artifact_size}})
        # Publishing is explicit. Test metrics are reported, not used as a hidden model-selection loop.
        if publish and not held_out_source:
            pending = settings.model_dir / 'hmeg.pending'
            import shutil
            shutil.copyfile(candidate, pending)
            os.replace(pending, settings.model_dir / 'hmeg.joblib')
        output = settings.model_dir / f'{version}.json'
        output.write_text(json.dumps({'training': manifest, 'experiments': results}, indent=2), encoding='utf-8')
        print(f'Training complete. Version {version}. Published: {publish and not held_out_source}', flush=True)
        return version
    finally:
        database.close()
