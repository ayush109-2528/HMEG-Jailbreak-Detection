from dataclasses import dataclass


@dataclass(frozen=True)
class Router:
    low: float = .2
    high: float = .8
    word_weight: float = .5

    def __post_init__(self):
        if not 0 <= self.low < self.high <= 1 or not 0 <= self.word_weight <= 1:
            raise ValueError('Invalid routing thresholds or weight.')

    def decide(self, scores: dict[str, float]) -> tuple[float, str]:
        weights = {'word': self.word_weight, 'format': 1 - self.word_weight}
        active = {key: value for key, value in scores.items() if key in weights}
        denominator = sum(weights[key] for key in active)
        if not active or denominator == 0:
            return .5, 'uncertain'
        fast = sum(weights[key] * value for key, value in active.items()) / denominator
        return fast, 'benign' if fast <= self.low else 'jailbreak' if fast >= self.high else 'uncertain'
