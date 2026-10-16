import math
import re
import unicodedata
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = ['character_count', 'word_count', 'approx_token_count', 'average_word_length',
                 'special_ratio', 'punctuation_ratio', 'digit_ratio', 'uppercase_ratio',
                 'whitespace_ratio', 'unicode_ratio', 'entropy', 'repeated_token_ratio',
                 'repeated_character_ratio', 'prefix_symbol_ratio', 'suffix_symbol_ratio']


def structure(text: str) -> dict:
    n = max(1, len(text))
    tokens = text.split()
    counts = Counter(text)
    ratio = lambda chars: sum(not c.isalnum() and not c.isspace() for c in chars) / max(1, len(chars))
    values = [len(text), len(tokens), len(re.findall(r'\w+|[^\w\s]', text)),
              sum(map(len, tokens)) / max(1, len(tokens)), ratio(text),
              sum(unicodedata.category(c).startswith('P') for c in text) / n,
              sum(c.isdigit() for c in text) / n, sum(c.isupper() for c in text) / n,
              sum(c.isspace() for c in text) / n, sum(ord(c) > 127 for c in text) / n,
              -sum((v/n)*math.log2(v/n) for v in counts.values()),
              1 - len(set(tokens)) / max(1, len(tokens)) if tokens else 0,
              sum(a == b for a, b in zip(text, text[1:])) / n,
              ratio(text[:64]), ratio(text[-64:])]
    return dict(zip(FEATURE_NAMES, values))


class StructuralFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, perplexity_provider=None):
        self.perplexity_provider = perplexity_provider

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            row = list(structure(text).values())
            if self.perplexity_provider is not None:
                row.append(float(self.perplexity_provider(text)))
            rows.append(row)
        return np.asarray(rows, dtype=float)
