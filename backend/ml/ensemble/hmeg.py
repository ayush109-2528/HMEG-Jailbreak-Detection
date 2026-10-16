"""Measured execution: no simulated probabilities and no silent missing features."""
from time import perf_counter
import numpy as np
from ml.preprocessing import preprocess
from .router import Router
from .flat_ensemble import fuse

KEYS = ('word', 'format', 'meaning', 'pattern', 'instruction')


class HMEG:
    def __init__(self, experts, meta, router=None, threshold=.5, version='unversioned'):
        self.experts, self.meta = experts, meta
        self.router = router or Router()
        self.threshold, self.version = threshold, version

    def predict(self, prompt: str, *, flat=False, fusion='meta_xgboost') -> dict:
        started = perf_counter()
        trace = [{'stage': 'Prompt received', 'at_ms': 0.0, 'key': 'input'}]
        mark = lambda stage, key: trace.append({'stage': stage, 'key': key, 'at_ms': (perf_counter() - started) * 1000})
        text = preprocess(prompt)
        mark('Preprocessing complete', 'preprocessing')
        scores, records = {}, {}

        def run(key):
            if key not in self.experts:
                records[key] = dict(executed=False, status='unavailable', score=None, latency_ms=None, details={})
                return
            before = perf_counter()
            scores[key] = self.experts[key].score(text)
            if not np.isfinite(scores[key]) or not 0 <= scores[key] <= 1:
                raise ValueError('Expert returned an invalid probability.')
            elapsed = (perf_counter() - before) * 1000
            records[key] = dict(executed=True, status='executed', score=scores[key], latency_ms=elapsed,
                                details=self.experts[key].inspect(text))
            mark(f'{key.title()} Expert complete', key)

        run('word')
        run('format')
        fast, route = self.router.decide(scores)
        mark('Router evaluated', 'router')
        layer1_time = (perf_counter() - started) * 1000
        deeper = flat or route == 'uncertain'
        before_deep = perf_counter()
        for key in KEYS[2:]:
            if deeper:
                run(key)
            else:
                records[key] = dict(executed=False, status='skipped', score=None, latency_ms=None, details={})
        if deeper:
            final = fuse(scores, fusion, self.meta)
            label = int(final >= self.threshold)
            mark('Meta classifier complete' if fusion.startswith('meta_') else 'Flat fusion complete', 'meta')
        else:
            final, label = fast, int(route == 'jailbreak')
        layer2_time = (perf_counter() - before_deep) * 1000 if deeper else 0.0
        mark('Final decision returned', 'decision')
        values = list(scores.values())
        variance = float(np.var(values))
        votes = sum(value >= .5 for value in values)
        return dict(prediction='jailbreak' if label else 'benign', label=label,
                    action='Block / flag prompt' if label else 'Allow prompt', final_probability=final,
                    decision_layer=2 if deeper else 1, model_version=self.version,
                    routing=dict(fast_score=fast, low_threshold=self.router.low, high_threshold=self.router.high,
                                 routed_to_layer_2=deeper, decision=route, final_threshold=self.threshold),
                    experts=records, experts_executed=len(scores), trace=trace,
                    total_latency_ms=(perf_counter() - started) * 1000,
                    layer1_time_ms=layer1_time, layer2_time_ms=layer2_time,
                    reduced_expert_set=len(self.experts) < 5,
                    agreement=dict(jailbreak_votes=votes, total=len(values), variance=variance,
                                   description='Strong Agreement' if votes in (0, len(values)) else
                                   'High Disagreement' if variance >= .1 else 'Mixed Signals'))
