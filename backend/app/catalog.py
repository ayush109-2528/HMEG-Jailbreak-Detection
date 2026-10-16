"""Public educational metadata; only explicitly listed source files are exposed."""
from pathlib import Path

EXPERTS = [
    dict(key='word', name='Word Expert', layer=1, purpose='Recognizes suspicious words and recurring phrases.', features='Word TF-IDF, unigrams and bigrams', classifier='Logistic Regression', limitation='Paraphrases can avoid familiar vocabulary.', file='ml/experts/word_expert.py'),
    dict(key='format', name='Format Expert', layer=1, purpose='Examines spelling, symbols, case, spacing, and character sequences.', features='Character 3–5 grams and structural ratios', classifier='Logistic Regression', limitation='Unusual but legitimate formatting can trigger false positives.', file='ml/experts/format_expert.py'),
    dict(key='meaning', name='Meaning Expert', layer=2, purpose='Represents sentence meaning using a frozen sentence encoder.', features='Quantized ONNX all-MiniLM-L6-v2 sentence embeddings', classifier='Logistic Regression', limitation='Requires the pinned encoder assets. Encodes up to 256 wordpieces; domain and language shifts affect it.', file='ml/experts/meaning_expert.py'),
    dict(key='pattern', name='Pattern Expert', layer=2, purpose='Examines abnormal structures and suffix-like patterns.', features='Length, entropy, repetition, prefix/suffix and character statistics', classifier='XGBoost', limitation='Structural anomalies alone do not establish malicious intent. Perplexity is optional and disabled by default.', file='ml/experts/pattern_expert.py'),
    dict(key='instruction', name='Instruction Expert', layer=2, purpose='Detects attempts to replace or manipulate instruction hierarchy.', features='Word TF-IDF 1–3 grams and hierarchy indicators', classifier='Logistic Regression', limitation='Quoted attacks in benign research discussions can be difficult to distinguish.', file='ml/experts/instruction_expert.py'),
]

TOPICS = [dict(key='preprocessing', name='Preprocessing', file='ml/preprocessing.py', explanation='Removes NUL characters and trims outer whitespace. Word features normalize case; format features preserve it.'),
    *[dict(key=x['key'], name=x['name'], file=x['file'], explanation=f"{x['purpose']} Features: {x['features']}. Classifier: {x['classifier']}. {x['limitation']}") for x in EXPERTS],
    dict(key='semantic_encoder', name='Semantic embedding runtime', file='ml/experts/semantic_encoder.py', explanation='Tokenizes up to 256 wordpieces, runs a frozen quantized MiniLM encoder on CPU, averages token embeddings with an attention mask, and normalizes the resulting vector. A separate Logistic Regression classifier learns the jailbreak labels.'),
    dict(key='router', name='Confidence router', file='ml/ensemble/router.py', explanation='Combines the two fast scores. Scores at or below the low threshold exit benign; scores at or above the high threshold exit jailbreak. The interval between them routes to Layer 2. Thresholds are experimental settings, not scientifically optimal constants.'),
    dict(key='meta', name='Learned fusion', file='ml/ensemble/meta_classifier.py', explanation='Fits a classifier on expert probabilities from a separate validation partition. Only available expert scores enter the vector, in a saved order. No absent score is replaced with zero.'),
    dict(key='hmeg', name='HMEG orchestration', file='ml/ensemble/hmeg.py', explanation='Runs the two fast experts, routes uncertainty, and conditionally executes the deeper experts. Records measured timings and explicitly distinguishes skipped and unavailable experts.'),
    dict(key='training', name='Training and evaluation', file='ml/training.py', explanation='Loads imported data, checks lineage and normalized-text overlap, fits experts on training rows, fits fusion on validation rows, and evaluates held-out test rows. Dataset limits and reduced expert sets are recorded.'),
    dict(key='metrics', name='Evaluation metrics', file='ml/evaluation/metrics.py', explanation='Calculates confusion counts, accuracy, precision, recall, F1, ROC-AUC, FPR, FNR and measured latency. AUC is unavailable when only one class is present.'),
    dict(key='mongodb', name='MongoDB storage', file='database/repositories.py', explanation='Reads paginated dataset records and saved experiments. Stores live prompts only when a visitor explicitly consents to public sharing. Credentials are never included in responses.')]


def code_topics():
    root = Path(__file__).resolve().parents[1]
    return [{**item, 'code': (root / item['file']).read_text(encoding='utf-8') if (root / item['file']).is_file() else 'Module pending implementation.'} for item in TOPICS]
