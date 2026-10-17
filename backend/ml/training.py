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


def load_data(db, max_train=20000, max_eval=1000, held_out_source=None, dataset_csv=None):
    projection = {'_id': 0, 'p': 1, 'l': 1, 'split': 1, 'src': 1, 'nh': 1, 'bid': 1, 'pid': 1, 'sid': 1, 'syn': 1}
    if dataset_csv:
        from scripts.import_dataset import records
        rows = ({key: value for key, value in row.items() if key in projection and key != '_id'} for row in records(dataset_csv))
    else:
        rows = db.dataset_samples.find({}, projection)
    frame = pd.DataFrame(list(rows))
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


def train(*, max_train=20000, max_eval=1000, semantic=False, publish=False, held_out_source=None, studies=False, latency_samples=0, dataset_csv=None, resume_run=None):
    started = time.perf_counter()
    database = Database(settings)
    try:
        if not database.connected():
            raise ValueError('MongoDB unavailable.')
        db = database.db
        meta = db.research_metadata.find_one({'key': 'dataset'})
        if not meta or meta['value'].get('status') != 'ready':
            raise ValueError('Dataset import must finish before training.')
        if dataset_csv:
            with Path(dataset_csv).open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest != meta['value']['sha256']:
                raise ValueError('Local CSV does not match the imported dataset checksum')
            print('Verified local CSV matches MongoDB dataset SHA-256.', flush=True)
        (train_set, val, test), manifest = load_data(db, max_train, max_eval, held_out_source, dataset_csv)
        manifest['training_source'] = 'Checksum-verified local CSV' if dataset_csv else 'MongoDB dataset_samples'

        print('Full split counts:', manifest['used_split_counts'], flush=True)
        for left, right in [(train_set, val), (train_set, test), (val, test)]:
            if set(left.group) & set(right.group) or set(left.nh) & set(right.nh):
                raise ValueError('Cross-partition leakage detected')
        manifest['latency_sample_limit'] = latency_samples
        manifest['semantic_threads'] = int(os.getenv('SEMANTIC_THREADS', '2'))
        version = resume_run or uuid4().hex
        if not all(c in "0123456789abcdef" for c in version) or len(version) != 32:
            raise ValueError("Invalid training run ID")
        checkpoint = settings.model_dir / 'training_runs' / version
        checkpoint.mkdir(parents=True, exist_ok=True)
        manifest['dataset_sha256'] = meta['value']['sha256']
        if semantic:
            from ml.experts.semantic_encoder import asset_dir
            manifest['semantic_encoder'] = json.loads((asset_dir() / 'manifest.json').read_text())
        identity = {key: manifest[key] for key in ['dataset_sha256', 'used_split_counts', 'max_train', 'max_eval', 'held_out_source']}
        identity['semantic'] = semantic
        identity_path = checkpoint / 'input_identity.json'
        if resume_run:
            if not identity_path.exists() or json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Resume checkpoint does not match dataset and split configuration')
        identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        experts = {}
        for key, cls in CLASSES.items():
            if key == 'meaning' and not semantic:
                continue
            if resume_run and (checkpoint / (key + '.joblib')).exists():
                experts[key] = joblib.load(checkpoint / (key + '.joblib'))
                print(f'Restored completed expert: {key}', flush=True)
                continue
            print(f'Training {key} expert on {len(train_set):,} rows', flush=True)
            experts[key] = cls().fit(train_set.p, train_set.l)
            joblib.dump(experts[key], checkpoint / (key + '.joblib'), compress=3)
            print(f'Checkpoint saved: {key}', flush=True)
        print('Fitting learned fusion on validation partition', flush=True)
        columns = {key: expert.probabilities(val.p) for key, expert in experts.items()}
        rows = [{key: float(value[i]) for key, value in columns.items()} for i in range(len(val))]
        metas = {kind: MetaClassifier(list(experts), kind).fit(rows, val.l.tolist()) for kind in ['logistic', 'xgboost']}
        manifest.update(available_experts=list(experts), reduced_expert_set=len(experts) < 5,
            training_seconds=time.perf_counter() - started, created_at=datetime.now(timezone.utc).isoformat())
        bundle = dict(experts=experts, metas=metas, preferred_meta='xgboost', version=version, training=manifest)
        engine = HMEG(experts, metas['xgboost'], Router(settings.low, settings.high), settings.threshold, version)
        results = []
        # Save fitted weights before lengthy evaluation, without promoting them to inference.
        joblib.dump(bundle, checkpoint / 'fitted.joblib', compress=3)
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        score_cache = None
        if latency_samples:
            from ml.evaluation.full_dataset import score_test_set, evaluate_full
            score_cache = score_test_set(experts, test)
            joblib.dump(score_cache, checkpoint / 'test_scores.joblib', compress=3)

        def measure(model, **options):
            if score_cache is not None:
                return evaluate_full(model, test, score_cache, latency_samples, **options)
            return evaluate(model, test, **options)


        def save(name, kind, measurement, extra=None):
            row = dict(experiment_id=uuid4().hex, name=name, model_type='ood' if held_out_source else kind,
                model_version=version, created_at=manifest['created_at'], dataset_sources=manifest['sources'],
                features={key: expert.method for key, expert in experts.items()}, training=manifest,
                routing_thresholds={'low': settings.low, 'high': settings.high, 'final': settings.threshold},
                hyperparameters={'seed': 42, 'meta': 'xgboost', 'expert_parameters': {key: str(expert.model.get_params(deep=False)) for key, expert in experts.items()}},
                training_time=manifest['training_seconds'], **measurement)
            row.update(extra or {})
            (checkpoint / (row['experiment_id'] + '.json')).write_text(json.dumps(row), encoding='utf-8')
            db.experiments.insert_one(dict(row))
            print('Evaluation saved:', name, flush=True)
            results.append(row)

        save('HMEG' + (' (four-expert configuration)' if len(experts) < 5 else ''), 'hmeg', measure(engine))
        for key, expert in experts.items():
            timings, probabilities = [], []
            if score_cache is not None:
                from ml.preprocessing import preprocess
                probabilities = [score_cache[key]['scores'][preprocess(text)] for text in test.p]
                timings = score_cache[key]['timings']
            else:
                for text in test.p:
                    before = time.perf_counter()
                    probabilities.append(expert.score(text))
                    timings.append((time.perf_counter() - before) * 1000)
            save(key.title() + ' Expert', 'expert', summarize(test.l.tolist(), [int(x >= .5) for x in probabilities], probabilities, timings))
        for method in ['majority', 'average', 'weighted', 'meta_logistic', 'meta_xgboost']:
            candidate = HMEG(experts, metas['logistic' if method == 'meta_logistic' else 'xgboost'], version=version)
            save('Flat ' + method.replace('_', ' ').title(), 'flat', measure(candidate, flat=True, fusion=method))
        if studies:
            for key, subset in without_each(experts):
                meta_model = MetaClassifier(list(subset)).fit([{k: r[k] for k in subset} for r in rows], val.l.tolist())
                save('HMEG without ' + key.title(), 'ablation', measure(HMEG(subset, meta_model)))
            for router in routers():
                low, high = router.low, router.high
                save(f'Routing {low} / {high}', 'routing', measure(HMEG(experts, metas['xgboost'], Router(low, high))), {'routing_thresholds': {'low': low, 'high': high, 'final': .5}})
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
