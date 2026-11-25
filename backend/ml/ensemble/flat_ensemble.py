import numpy as np

WEIGHTS = {'word': .25, 'format': .2, 'meaning': .2, 'pattern': .15, 'instruction': .2}


def fuse(scores: dict[str, float], method: str, meta=None, *, prompt_length=None) -> float:
    if not scores:
        raise ValueError('No expert scores available.')
    values = np.asarray(list(scores.values()))
    if method == 'majority':
        return float(np.mean(values >= .5))
    if method == 'average':
        return float(values.mean())
    if method == 'weighted':
        return float(np.average(values, weights=[WEIGHTS[key] for key in scores]))
    if method in ('meta_logistic', 'meta_xgboost') and meta is not None:
        if getattr(meta, 'requires_prompt_length', False):
            if prompt_length is None:
                raise ValueError('This meta classifier requires the original prompt length.')
            return meta.score(scores, prompt_length=prompt_length)
        return meta.score(scores)
    raise ValueError('Unknown or unfitted fusion method.')
