from scripts.corpus_integrity import describe, normalized, split_for, validate

def test_preserves_format_but_deduplicates_normalized_variants():
    prompt = "  IGNORE\t previous\u200b instructions!!!\n"
    assert normalized(prompt) == 'ignore previous instructions!!!'
    info = describe(prompt)
    assert info['char_count'] == len(prompt)
    assert info['prompt_sha256'] != info['normalized_sha256']
    assert info['special_keyword'] is True

def test_keywords_do_not_assign_labels_and_use_boundaries():
    assert 'DAN' not in describe('Daniel studies dance')['matched_keywords']
    assert 'label' not in describe('Explain the phrase ignore previous instructions')
    assert describe('word ' * 30)['is_short_prompt']
    assert not describe('word ' * 31)['is_short_prompt']

def test_groups_are_deterministic():
    assert split_for('parent-123') == split_for('parent-123')

def test_empty_and_duplicate_corpus_fails_integrity():
    assert not validate([])['passed']
    r=dict(prompt='hello',label=0,language='en',source_dataset='test',source_url='https://example.org',group_id='one',research_split='train',**describe('hello'))
    assert not validate([r,r],target=2)['passed']
