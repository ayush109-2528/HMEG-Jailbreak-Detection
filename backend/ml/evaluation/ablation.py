def without_each(experts):
    """Generate actual available-expert ablations; never invent a missing expert."""
    for key in experts:
        yield key, {name: expert for name, expert in experts.items() if name != key}
