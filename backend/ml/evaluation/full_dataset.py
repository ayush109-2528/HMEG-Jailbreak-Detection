"""Full-test expert scores; separately sampled real end-to-end latency. No estimated latency."""
import numpy as np
from time import perf_counter
from ml.preprocessing import preprocess
from ml.ensemble.hmeg import HMEG
from ml.evaluation.metrics import summarize


class CachedExpert:
    def __init__(self, scores):
        self.scores = scores
    def score(self, prompt):
        return self.scores[prompt]
    def inspect(self, prompt):
        return {}


def score_test_set(experts, test):
    cache = {}
    for key, expert in experts.items():
        scores, timings = {}, []
        for index, text in enumerate(test.p):
            text = preprocess(text)
            start = perf_counter()
            value = expert.score(text)
            timings.append((perf_counter() - start) * 1000)
            scores[text] = value
            if index % 2000 == 0 or index + 1 == len(test):
                print(f"Test scores {key}: {index + 1:,}/{len(test):,}", flush=True)
        cache[key] = {'scores': scores, 'timings': timings}
    return cache


def evaluate_full(engine, test, cache, latency_samples, *, flat=False, fusion='meta_xgboost'):
    cached = HMEG({key: CachedExpert(cache[key]['scores']) for key in engine.experts},
                  engine.meta, engine.router, engine.threshold, engine.version)
    # Only scores, labels and routing counts come from cached execution. Its timers are discarded.
    results = [cached.predict(text, flat=flat, fusion=fusion) for text in test.p]
    indices = np.sort(np.random.default_rng(42).choice(len(test), min(latency_samples, len(test)), replace=False))
    measured = [engine.predict(test.p.iloc[int(i)], flat=flat, fusion=fusion) for i in indices]
    out = summarize(test.l.tolist(), [r['label'] for r in results],
                    [r['final_probability'] for r in results], [r['total_latency_ms'] for r in measured])
    out.update(classification_sample_count=len(test), latency_sample_count=len(measured),
               evaluation_method='All test rows; real per-prompt expert scores reused for fusion. End-to-end latency measured separately on seeded test sample.',
               layer1_exit_rate=float(np.mean([r['decision_layer'] == 1 for r in results])),
               layer2_routing_rate=float(np.mean([r['routing']['routed_to_layer_2'] for r in results])),
               average_experts_executed=float(np.mean([r['experts_executed'] for r in results])),
               average_layer1_time=float(np.mean([r['layer1_time_ms'] for r in measured])),
               average_layer2_time=float(np.mean([r['layer2_time_ms'] for r in measured])))
    return out
