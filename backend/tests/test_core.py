"""Isolated control-flow checks. Fixtures are not research data or model results."""
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from app.schemas import PredictionInput
from ml.ensemble.router import Router
from ml.ensemble.hmeg import HMEG, KEYS
from ml.utils.model_info import ModelStore
from ml.evaluation.metrics import summarize


class StubExpert:
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0

    def score(self, prompt):
        self.calls += 1
        return self.probability

    def inspect(self, prompt):
        return {}


class StubMeta:
    def __init__(self, keys):
        self.keys = keys

    def score(self, scores):
        assert set(scores) == set(self.keys)
        return .7


@pytest.mark.parametrize('score,decision', [(.2, 'benign'), (.8, 'jailbreak'), (.5, 'uncertain')])
def test_router_boundaries(score, decision):
    assert Router().decide({'word': score, 'format': score}) == (score, decision)


def test_router_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        Router(.8, .2)


@pytest.mark.parametrize('score,label', [(.1, 0), (.9, 1)])
def test_early_exit_does_not_execute_deeper_experts(score, label):
    experts = {key: StubExpert(score) for key in KEYS}
    result = HMEG(experts, StubMeta(KEYS)).predict('test fixture')
    assert result['label'] == label
    assert result['decision_layer'] == 1
    assert result['experts_executed'] == 2
    assert all(experts[key].calls == 0 for key in KEYS[2:])
    assert all(result['experts'][key]['score'] is None for key in KEYS[2:])
    assert result['layer2_time_ms'] == 0


def test_uncertain_executes_all_experts_and_meta():
    experts = {key: StubExpert(.5) for key in KEYS}
    result = HMEG(experts, StubMeta(KEYS)).predict('test fixture')
    assert result['decision_layer'] == 2
    assert result['experts_executed'] == 5
    assert result['final_probability'] == .7
    assert all(expert.calls == 1 for expert in experts.values())


def test_reduced_expert_set_has_no_fabricated_semantic_score():
    keys = [key for key in KEYS if key != 'meaning']
    result = HMEG({key: StubExpert(.5) for key in keys}, StubMeta(keys)).predict('test fixture')
    assert result['reduced_expert_set'] is True
    assert result['experts']['meaning']['status'] == 'unavailable'
    assert result['experts']['meaning']['score'] is None
    assert result['experts_executed'] == 4


def test_missing_artifact_does_not_invent_predictions(tmp_path):
    store = ModelStore(SimpleNamespace(model_dir=tmp_path))
    assert store.engine is None
    assert store.status()['models_loaded'] is False


def test_prompt_validation_and_consent_default():
    assert PredictionInput(prompt='test fixture').public_consent is False
    for text in ('', '   ', 'x' * 10001):
        with pytest.raises(ValidationError):
            PredictionInput(prompt=text)


def test_metrics_confusion_matrix_and_undefined_auc():
    result = summarize([0, 0, 1, 1], [0, 1, 0, 1], [.1, .7, .3, .9], [1, 2, 3, 4])
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert result['metrics']['fpr'] == .5
    assert result['metrics']['fnr'] == .5
    assert summarize([0, 0], [0, 0], [.1, .2], [1, 2])['metrics']['roc_auc'] is None


class RecordingExpert(StubExpert):
    def score(self, prompt):
        self.last_prompt = prompt
        return super().score(prompt)


def test_new_models_preserve_format_without_changing_legacy_defaults():
    prompt = '  Example\ttext.\n'
    experts = {key: RecordingExpert(.1) for key in KEYS}
    HMEG(experts, StubMeta(KEYS), preserve_format=True).predict(prompt)
    assert all(expert.last_prompt == prompt for expert in experts.values() if expert.calls)
    HMEG(experts, StubMeta(KEYS)).predict(prompt)
    assert experts['format'].last_prompt == prompt.strip()


def test_candidate_bundle_loads_recorded_routing_and_format_policy(tmp_path):
    import joblib
    experts = {'word': RecordingExpert(.1), 'format': RecordingExpert(.1)}
    joblib.dump({'experts':experts,'metas':{'logistic':StubMeta(experts)},
                'preferred_meta':'logistic','version':'test-fixture',
                'routing_thresholds':{'low':.1,'high':.9,'final':.5},
                'preserve_format':True}, tmp_path/'hmeg.joblib')
    config = SimpleNamespace(model_dir=tmp_path,low=.2,high=.8,threshold=.5)
    model = ModelStore(config)
    assert model.engine is not None
    assert model.engine.router.low == .1 and model.engine.router.high == .9
    assert model.engine.preserve_format is True
