"""Public research reads and HMEG inference. Training is a separate operator worker."""
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock, Semaphore
from time import monotonic
import os
import platform
import importlib.metadata
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from app.config import settings
from app.schemas import PredictionInput
from app.prediction_policy import apply_prediction_policy
from app.catalog import EXPERTS, code_topics
from database.mongodb import Database
from database.repositories import Repository
from ml.utils.model_info import ModelStore


@asynccontextmanager
async def lifespan(app):
    app.state.database = Database(settings)
    app.state.repo = Repository(app.state.database)
    app.state.models = ModelStore(settings)
    app.state.model_mtime = artifact_mtime()
    yield
    app.state.database.close()


def artifact_mtime():
    path = settings.model_dir / 'hmeg.joblib'
    return path.stat().st_mtime_ns if path.is_file() else None


app = FastAPI(title='HMEG Research API', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.origins), allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
slots = Semaphore(2)
reload_lock = Lock()
rate_lock = Lock()
visits = defaultdict(deque)


@app.exception_handler(PyMongoError)
@app.exception_handler(ConnectionError)
def database_error(request, error):
    return JSONResponse(status_code=503, content={'detail': 'Database unavailable. Retry shortly or check the backend configuration.'})


def store(request):
    with reload_lock:
        stamp = artifact_mtime()
        if stamp != request.app.state.model_mtime:
            request.app.state.models = ModelStore(settings)
            request.app.state.model_mtime = stamp
    return request.app.state.models


@app.get('/api/health')
def health(request: Request):
    return {'status': 'ok', 'database_connected': request.app.state.database.connected(), **store(request).status()}


@app.get('/api/dataset/stats')
def stats(request: Request):
    return request.app.state.repo.stats()


@app.get('/api/dataset/samples')
def samples(request: Request, page: int = Query(1, ge=1, le=100000), limit: int = Query(25, ge=1, le=100),
            search: str = Query('', max_length=200), label: int | None = Query(None, ge=0, le=1),
            source_dataset: str | None = None, attack_category: str | None = None, expert_focus: str | None = None,
            research_split: str | None = None, is_synthetic: bool | None = None, verification_status: str | None = None,
            transformation: str | None = None):
    return request.app.state.repo.samples(page, limit, search, label=label, source_dataset=source_dataset,
        attack_category=attack_category, expert_focus=expert_focus, research_split=research_split,
        is_synthetic=is_synthetic, verification_status=verification_status, transformation=transformation)


@app.get('/api/dataset/sources')
def sources(request: Request):
    return request.app.state.repo.sources()


@app.get('/api/database/stats')
def database_stats(request: Request):
    return request.app.state.repo.database_stats()


@app.get('/api/experts')
def experts(request: Request):
    model = store(request)
    return [{**item, 'status': 'loaded' if model.bundle and item['key'] in model.bundle['experts'] else 'unavailable'} for item in EXPERTS]


@app.get('/api/experiments')
def experiments(request: Request, model_type: str | None = None, search: str = Query('', max_length=200)):
    return request.app.state.repo.experiments(model_type, search)


@app.get('/api/experiments/{identifier}')
def experiment(identifier: str, request: Request):
    row = request.app.state.repo.experiment(identifier)
    if not row:
        raise HTTPException(404, 'Experiment not found')
    return row


@app.get('/api/code')
def code():
    return code_topics()


@app.get('/api/routing/config')
def routing():
    return {'low': settings.low, 'high': settings.high, 'final_threshold': settings.threshold, 'note': 'Operator-configured, tunable experimental thresholds.'}


def installed_version(package):
    # Linux's small CPU wheel exports xgboost but registers as xgboost-cpu.
    distributions = (package, 'xgboost-cpu') if package == 'xgboost' else (package,)
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@app.get('/api/reproducibility')
def reproducibility(request: Request):
    model = store(request)
    try:
        dataset = request.app.state.repo.metadata('dataset')
    except (ConnectionError, PyMongoError):
        dataset = None
    return {'dataset': dataset, 'model': model.status(), 'routing': routing(), 'random_seed': 42,
        'git_commit': os.getenv('GIT_COMMIT') or None, 'python': platform.python_version(),
        'versions': {package: installed_version(package) for package in ['scikit-learn', 'numpy', 'pandas', 'xgboost', 'pymongo', 'fastapi', 'onnxruntime', 'tokenizers']},
        'training': model.bundle.get('training') if model.bundle else None}


@app.get('/api/live-tests')
def live_tests(request: Request):
    return request.app.state.repo.public_tests()


@app.get('/api/live-tests/{identifier}')
def live_test(identifier: str, request: Request):
    row = request.app.state.repo.public_test(identifier)
    if not row:
        raise HTTPException(404, 'Public result not found')
    return row


@app.post('/api/predict/hmeg')
def predict(payload: PredictionInput, request: Request):
    now, key = monotonic(), request.client.host if request.client else 'unknown'
    with rate_lock:
        for old in list(visits):
            while visits[old] and visits[old][0] < now - 60:
                visits[old].popleft()
            if not visits[old]:
                del visits[old]
        if len(visits[key]) >= settings.requests_per_minute:
            raise HTTPException(429, 'Too many analyses. Try again in one minute.')
        visits[key].append(now)
    model = store(request)
    if model.engine is None:
        raise HTTPException(503, 'Trained HMEG model unavailable. No simulated prediction is substituted.')
    if not slots.acquire(blocking=False):
        raise HTTPException(429, 'The inference workers are busy. Please retry shortly.')
    try:
        result = apply_prediction_policy(payload.prompt, model.engine.predict(payload.prompt))
    except Exception:
        raise HTTPException(503, 'Model inference failed. Check artifact and dependency compatibility.')
    finally:
        slots.release()
    result['saved'] = False
    if payload.public_consent:
        try:
            result['result_id'] = request.app.state.repo.save_public_test(payload.prompt, dict(result))
            result['saved'] = True
        except (ConnectionError, PyMongoError):
            result['storage_warning'] = 'Prediction succeeded, but public sharing could not be saved.'
    return result
