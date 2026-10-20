"""Serving-time abstention, separate from the evaluated binary research model.

Isolated alphabetic tokens do not provide enough instruction context for this
model. Never relabel them as benign or reinterpret the learned probability.
"""
import unicodedata

POLICY_VERSION = "isolated-token-v1"


def isolated_token(prompt: str) -> bool:
    text = prompt.strip().strip(".!,?;:\'\"()[]{}")
    return bool(text) and any(c.isalpha() for c in text) and all(
        unicodedata.category(c)[0] in ('L', 'M') for c in text)


def apply_prediction_policy(prompt: str, result: dict) -> dict:
    output = dict(result)
    output['decision_policy'] = POLICY_VERSION
    output['abstained'] = isolated_token(prompt)
    if output['abstained']:
        output['raw_model'] = {key: result[key] for key in
                               ('prediction', 'label', 'final_probability', 'action')}
        output.update(prediction='insufficient_context', label=None,
                      final_probability=None, action='Request more context / manual review',
                      decision_note='An isolated alphabetic token provides insufficient instruction context. '
                      'The current model overflags some single words and random letters. '
                      'No benign or jailbreak decision is issued; provide a complete prompt. '
                      'Raw expert scores remain available for research inspection.')
    return output
