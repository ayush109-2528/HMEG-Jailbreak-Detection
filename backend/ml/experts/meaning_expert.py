from sklearn.linear_model import LogisticRegression
from .base import Expert


class MeaningExpert(Expert):
    method = 'Quantized MiniLM-L6-v2 ONNX sentence embeddings + Logistic Regression'

    def __init__(self, encoder_path='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder_path = encoder_path
        self._encoder = None
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

    def encoder(self):
        if self._encoder is None:
            from .semantic_encoder import SemanticEncoder
            self._encoder = SemanticEncoder()
        return self._encoder

    def fit(self, prompts, labels):
        from time import perf_counter
        start = perf_counter()
        from app.config import settings
        from .semantic_encoder import asset_dir
        from ml.training_embeddings import cached_embeddings
        vectors = cached_embeddings(self.encoder(), list(prompts),
            settings.model_dir / 'embedding_cache', (asset_dir() / 'manifest.json').read_bytes())
        self.model.fit(vectors, list(labels))
        self.training_time = perf_counter() - start
        return self

    def probabilities(self, prompts):
        vectors = self.encoder().encode(list(prompts), show_progress_bar=False)
        return self.model.predict_proba(vectors)[:, list(self.model.classes_).index(1)]

    def __getstate__(self):
        return {**self.__dict__, '_encoder': None}

    def inspect(self, prompt):
        return {"method": self.method, "encoder": "all-MiniLM-L6-v2",
                "runtime": "quantized ONNX CPU", "max_tokens": 256,
                "note": "Only the first 256 wordpieces are encoded; the frozen encoder is not fine-tuned."}
