COLLECTIONS = ('dataset_samples', 'dataset_sources', 'experiments', 'live_tests', 'research_metadata')
# Derive class_name from label; retain lineage and hashes for reproducibility.
FIELDS = {'sample_id': 'sid', 'prompt': 'p', 'label': 'l', 'attack_category': 'ac',
          'expert_focus': 'ef', 'source_dataset': 'src', 'source_record_id': 'srid',
          'base_record_id': 'bid', 'provenance': 'prov', 'is_synthetic': 'syn',
          'transformation': 'tr', 'parent_id': 'pid', 'research_split': 'split',
          'verification_status': 'ver', 'prompt_sha256': 'ph', 'normalized_sha256': 'nh',
          'class_name': 'cls', 'language': 'lang', 'word_count': 'wc', 'char_count': 'cc',
          'is_short_prompt': 'short', 'special_keyword': 'kw', 'matched_keywords': 'keywords',
          'attack_type': 'at', 'attack_family': 'af', 'aggregation_source': 'agg',
          'source_url': 'url', 'license': 'license', 'group_id': 'gid',
          'source_category': 'source_category', 'source_revision': 'source_revision'}


def readable(document: dict) -> dict:
    result = {name: document.get(short, document.get(name)) for name, short in FIELDS.items()}
    result['class_name'] = result.get('class_name') or ('jailbreak' if result['label'] == 1 else 'benign')
    result['attack_category'] = result.get('attack_category') or result.get('attack_family')
    return result
