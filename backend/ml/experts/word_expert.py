from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ml.preprocessing import words
from .base import Expert


class WordExpert(Expert):
    method = 'Word TF-IDF (1–2 grams) + Logistic Regression'

    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(preprocessor=words, ngram_range=(1, 2), max_features=20000)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ])

    def inspect(self, prompt: str) -> dict:
        vectorizer = self.model.named_steps['tfidf']
        row = vectorizer.transform([prompt]).tocoo()
        names = vectorizer.get_feature_names_out()
        terms = sorted(zip(row.col, row.data), key=lambda item: item[1], reverse=True)[:8]
        return {'recognized_terms': [{'term': str(names[i]), 'tfidf': float(v)} for i, v in terms],
                'note': 'Highest TF-IDF features, not causal explanations.'}
