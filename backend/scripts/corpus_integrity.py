"""Conservative, format-preserving corpus checks. No network/database mutations."""
import hashlib
import re
import unicodedata
from collections import Counter

KEYWORDS = (
    'ignore previous', 'ignore all previous', 'ignore instructions', 'forget previous',
    'disregard', 'override', 'system prompt', 'system message', 'developer message',
    'previous instructions', 'previous rules', 'act as', 'pretend to be', 'from now on',
    'roleplay', 'reveal instructions', 'show prompt', 'hidden instructions', 'bypass',
    'restrictions', 'jailbreak', 'do anything now', 'DAN', 'unrestricted', 'do not follow',
    'follow my instructions', 'priority instruction', 'repeat your instructions')
PATTERNS = [(word, re.compile(r'(?<!\w)' + re.escape(word) + r'(?!\w)', re.I)) for word in KEYWORDS]

def normalized(prompt):
    value = unicodedata.normalize('NFKC', prompt)
    value = re.sub('[\u200b-\u200d\ufeff\u2060]', '', value)
    return ' '.join(value.split()).lower()

def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def describe(prompt):
    words = len(prompt.split())
    norm = normalized(prompt)
    matches = [word for word, pattern in PATTERNS if word.lower() in norm and pattern.search(norm)]
    return dict(word_count=words, char_count=len(prompt), is_short_prompt=words <= 30,
                special_keyword=bool(matches), matched_keywords=matches,
                prompt_sha256=sha(prompt), normalized_sha256=sha(norm))

def split_for(group):
    bucket = int(sha('42:' + group)[:16], 16) % 10
    return 'train' if bucket < 8 else 'validation' if bucket == 8 else 'test'

def validate(rows, target=275000):
    labels = Counter(r['label'] for r in rows)
    hashes = [r['normalized_sha256'] for r in rows]
    groups = {}
    leakage = set()
    for r in rows:
        group = r['group_id']
        if group in groups and groups[group] != r['research_split']:
            leakage.add(group)
        groups[group] = r['research_split']
    checks = {
        'row_count': len(rows) == target,
        'balanced_classes': labels == {0:target//2, 1:target//2},
        'zero_normalized_duplicates': len(set(hashes)) == len(rows),
        'no_empty_prompts': all(isinstance(r['prompt'],str) and r['prompt'].strip() for r in rows),
        'valid_languages': all(r['language'] in ('en','hi') for r in rows),
        'both_classes_in_each_split': all({r['label'] for r in rows if r['research_split']==part}=={0,1} for part in ('train','validation','test')),
        'valid_splits': all(r['research_split'] in ('train','validation','test') for r in rows),
        'source_populated': all(r['source_dataset'] and r['source_url'] for r in rows),
        'hashes_match_original': all(r['prompt_sha256']==sha(r['prompt']) and r['normalized_sha256']==sha(normalized(r['prompt'])) for r in rows),
        'no_group_leakage': not leakage,
    }
    return dict(passed=all(checks.values()), checks=checks, total_rows=len(rows),
                class_counts=dict(labels), normalized_duplicates=len(rows)-len(set(hashes)),
                leaking_groups=len(leakage))
