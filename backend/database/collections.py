COLLECTIONS = ('dataset_samples', 'dataset_sources', 'experiments', 'live_tests', 'research_metadata')
# Derive class_name from label; retain lineage and hashes for reproducibility.
FIELDS = {'sample_id': 'sid', 'prompt': 'p', 'label': 'l', 'attack_category': 'ac',
          'expert_focus': 'ef', 'source_dataset': 'src', 'source_record_id': 'srid',
          'base_record_id': 'bid', 'provenance': 'prov', 'is_synthetic': 'syn',
          'transformation': 'tr', 'parent_id': 'pid', 'research_split': 'split',
          'verification_status': 'ver', 'prompt_sha256': 'ph', 'normalized_sha256': 'nh'}


def readable(document: dict) -> dict:
    result = {name: document.get(short) for name, short in FIELDS.items()}
    result['class_name'] = 'jailbreak' if result['label'] == 1 else 'benign'
    return result
