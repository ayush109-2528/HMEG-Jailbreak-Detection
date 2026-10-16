def create_indexes(db) -> None:
    """Run via CLI only. Keep indexes small and intentional."""
    db.dataset_samples.create_index('sid', unique=True)
    for field in ('l', 'src', 'split', 'syn', 'ac'):
        db.dataset_samples.create_index(field)
    db.dataset_sources.create_index('dataset_name', unique=True)
    db.experiments.create_index('experiment_id', unique=True)
    db.experiments.create_index('created_at')
    db.live_tests.create_index('result_id', unique=True)
    db.live_tests.create_index('timestamp')
    db.research_metadata.create_index('key', unique=True)
