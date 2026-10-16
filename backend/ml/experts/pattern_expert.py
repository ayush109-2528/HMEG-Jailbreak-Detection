from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .base import Expert
from .features import StructuralFeatures, structure


class PatternExpert(Expert):
    method = 'Length, entropy, repetition, ratios and prefix/suffix statistics + XGBoost'

    def __init__(self, perplexity_provider=None):
        self.model = Pipeline([
            ('features', StructuralFeatures(perplexity_provider)),
            ('scale', StandardScaler()),
            ('classifier', XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.08,
                                         n_jobs=1, random_state=42, eval_metric='logloss')),
        ])

    def inspect(self, prompt: str) -> dict:
        return {**structure(prompt), 'perplexity': 'Not configured'}
