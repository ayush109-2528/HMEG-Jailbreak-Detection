import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml.preprocessing import words
from .base import Expert

PATTERNS = ('ignore previous', 'system prompt', 'developer instruction', 'replace your',
            'override', 'act as', 'pretend', 'forget your', 'new instructions', 'disable rules')


class InstructionIndicators(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([[float(term in words(text)) for term in PATTERNS] for text in X])


class InstructionExpert(Expert):
    method = 'Word 1–3 gram TF-IDF + instruction hierarchy indicators + Logistic Regression'

    def __init__(self):
        self.model = Pipeline([
            ('features', FeatureUnion([
                ('text', TfidfVectorizer(preprocessor=words, ngram_range=(1, 3), max_features=20000)),
                ('hierarchy', InstructionIndicators()),
            ])),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ])

    def inspect(self, prompt: str) -> dict:
        return {'matched_indicators': [p for p in PATTERNS if p in words(prompt)],
                'note': 'Indicators are features, not proof of malicious intent.'}
