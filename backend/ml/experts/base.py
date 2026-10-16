from time import perf_counter


class Expert:
    def fit(self, prompts, labels):
        started = perf_counter()
        self.model.fit(list(prompts), list(labels))
        self.training_time = perf_counter() - started
        return self

    def probabilities(self, prompts):
        return self.model.predict_proba(list(prompts))[:, list(self.model.classes_).index(1)]

    def score(self, prompt: str) -> float:
        return float(self.probabilities([prompt])[0])

    def inspect(self, prompt: str) -> dict:
        return {'method': self.method}
