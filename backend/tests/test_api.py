import mongomock
import pytest
from fastapi.testclient import TestClient
from app import main
from app.config import Settings
from ml.utils.model_info import ModelStore


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = mongomock.MongoClient().hmeg_test
    class TestDatabase:
        def __init__(self, config): self.db = db
        def connected(self): return True
        def close(self): pass
    monkeypatch.setattr(main, 'Database', TestDatabase)
    monkeypatch.setattr(main, 'ModelStore', lambda config: ModelStore(Settings(model_dir=tmp_path)))
    monkeypatch.setattr(main, 'artifact_mtime', lambda: None)
    main.visits.clear()
    with TestClient(main.app) as client:
        yield client


def test_health_without_models(client):
    row = client.get('/api/health').json()
    assert row['status'] == 'ok'
    assert row['database_connected'] is True
    assert row['models_loaded'] is False


def test_no_mock_predictions(client):
    response = client.post('/api/predict/hmeg', json={'prompt': 'Explain rainbows'})
    assert response.status_code == 503
    assert 'simulated' in response.json()['detail']


def test_invalid_prompt(client):
    assert client.post('/api/predict/hmeg', json={'prompt': '  '}).status_code == 422
    assert client.post('/api/predict/hmeg', json={'prompt': 'x' * 10001}).status_code == 422


def test_code_whitelist(client):
    response = client.get('/api/code')
    assert response.status_code == 200
    assert len(response.json()) == 12
    assert all('.env' not in row['file'] for row in response.json())
    assert all('pending implementation' not in row['code'] for row in response.json())


def test_public_site_cannot_train_or_change_configuration(client):
    assert client.post('/api/train/hmeg').status_code == 404
    assert client.put('/api/routing/config', json={'low': .1}).status_code == 405
    assert client.get('/api/live-tests/not-a-result').status_code == 404


def test_pagination_bounds(client):
    assert client.get('/api/dataset/samples?limit=10000').status_code == 422
    assert client.get('/api/dataset/samples?page=0').status_code == 422


def test_no_consent_means_no_storage(client):
    class Engine:
        def predict(self, prompt): return {'label': 0}
    client.app.state.models.engine = Engine()
    response = client.post('/api/predict/hmeg', json={'prompt': 'test'})
    assert response.status_code == 200
    assert response.json()['saved'] is False
    assert client.app.state.database.db.live_tests.count_documents({}) == 0


def test_public_consent_creates_readable_result(client):
    class Engine:
        def predict(self, prompt): return {'label': 0}
    client.app.state.models.engine = Engine()
    response = client.post('/api/predict/hmeg', json={'prompt': 'test consent fixture', 'public_consent': True})
    assert response.json()['saved'] is True
    identifier = response.json()['result_id']
    assert client.get('/api/live-tests/' + identifier).json()['prompt'] == 'test consent fixture'
