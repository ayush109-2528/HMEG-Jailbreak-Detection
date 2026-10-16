import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class MetaClassifier:
    def __init__(self, keys: list[str], kind='xgboost'):
        self.keys, self.kind = list(keys), kind
        self.model = (XGBClassifier(n_estimators=80, max_depth=3, learning_rate=.08, n_jobs=1,
                                   random_state=42, eval_metric='logloss') if kind == 'xgboost'
                      else LogisticRegression(max_iter=1000, random_state=42))

    def matrix(self, rows):
        return np.asarray([[row[key] for key in self.keys] for row in rows], dtype=float)

    def fit(self, rows, labels):
        self.model.fit(self.matrix(rows), labels)
        return self

    def score(self, scores: dict[str, float]) -> float:
        return float(self.model.predict_proba(self.matrix([scores]))[0, list(self.model.classes_).index(1)])
