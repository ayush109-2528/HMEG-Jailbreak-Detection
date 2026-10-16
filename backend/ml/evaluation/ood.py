def source_holdout(frame, held_out: str, training_sources: list[str] | None = None):
    """Select an unseen source and remove shared parents and normalized text."""
    if training_sources and held_out in training_sources:
        raise ValueError('Held-out source cannot also be a training source.')
    test = frame[frame.src == held_out].copy()
    remaining = frame[frame.src != held_out].copy()
    if training_sources:
        remaining = remaining[remaining.src.isin(training_sources)]
    remaining = remaining[~remaining.group.isin(test.group) & ~remaining.nh.isin(test.nh)]
    if test.empty or remaining.empty:
        raise ValueError('Source selection must leave nonempty training and held-out sets.')
    return remaining, test
