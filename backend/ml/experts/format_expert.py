from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .base import Expert
from .features import StructuralFeatures, structure


class FormatExpert(Expert):
    method = 'Case-sensitive character 3–5 grams + formatting statistics + Logistic Regression'

    def __init__(self):
        self.model = Pipeline([
            ('features', FeatureUnion([
                ('characters', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False, max_features=18000)),
                ('format', Pipeline([('extract', StructuralFeatures()), ('scale', StandardScaler())])),
            ])),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ])

    def inspect(self, prompt: str) -> dict:
        return structure(prompt)
