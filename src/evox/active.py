from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

H = TypeVar("H")
I = TypeVar("I", bound=Hashable)


class PredictiveDistribution(Protocol):
    def log_prob(self, observation: object) -> float:
        ...

    def quadrature(self) -> tuple[tuple[object, float], ...]:
        ...


@dataclass(frozen=True)
class CategoricalDistribution:
    probabilities: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        probs = {key: float(value) for key, value in self.probabilities.items()}
        if not probs:
            raise ValueError("categorical distribution must have at least one outcome")
        if any((not math.isfinite(value)) or value < 0.0 for value in probs.values()):
            raise ValueError("categorical probabilities must be finite and non-negative")
        total = float(sum(probs.values()))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("categorical probabilities must sum to 1")
        if total <= 0.0:
            raise ValueError("categorical distribution must have positive mass")
        object.__setattr__(self, "probabilities", probs)

    def log_prob(self, observation: object) -> float:
        probability = float(self.probabilities.get(observation, 0.0))
        if probability <= 0.0:
            return float("-inf")
        return math.log(probability)

    def quadrature(self) -> tuple[tuple[object, float], ...]:
        return tuple(
            (outcome, float(self.probabilities[outcome]))
            for outcome in sorted(self.probabilities, key=repr)
            if self.probabilities[outcome] > 0.0
        )


@dataclass(frozen=True)
class GaussianDistribution:
    mean: float
    sigma: float
    quadrature_points: int = 21

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mean)):
            raise ValueError("Gaussian mean must be finite")
        if not math.isfinite(float(self.sigma)) or self.sigma <= 0.0:
            raise ValueError("Gaussian sigma must be finite and positive")
        if self.quadrature_points < 3:
            raise ValueError("quadrature_points must be at least 3")

    def log_prob(self, observation: object) -> float:
        try:
            value = float(observation)
        except (TypeError, ValueError) as exc:
            raise TypeError("Gaussian observations must be scalar numeric values") from exc
        z = (value - float(self.mean)) / float(self.sigma)
        return -math.log(float(self.sigma) * math.sqrt(2.0 * math.pi)) - 0.5 * z * z

    def quadrature(self) -> tuple[tuple[object, float], ...]:
        nodes, weights = np.polynomial.hermite.hermgauss(self.quadrature_points)
        observations = float(self.mean) + math.sqrt(2.0) * float(self.sigma) * nodes
        normalized = weights / math.sqrt(math.pi)
        return tuple((float(x), float(w)) for x, w in zip(observations, normalized))


@dataclass(frozen=True)
class IdentificationStep(Generic[I]):
    intervention: I
    information_gain: float
    observation: object
    prior: np.ndarray
    posterior: np.ndarray
    confidence: float


@dataclass(frozen=True)
class IdentificationResult(Generic[H, I]):
    map_hypothesis: H
    posterior: np.ndarray
    steps: tuple[IdentificationStep[I], ...]
    reached_confidence: bool
    probes_used: int


def entropy_bits(probabilities: np.ndarray | Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    positive = values[values > 0.0]
    if len(positive) == 0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)))


def _normalize_prior(prior: np.ndarray | Sequence[float], n: int) -> np.ndarray:
    values = np.asarray(prior, dtype=float)
    if values.shape != (n,):
        raise ValueError(f"prior must have shape ({n},)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("prior probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("prior must contain positive mass")
    return values / total


def posterior_update(
    prior: np.ndarray | Sequence[float],
    predictions: Sequence[PredictiveDistribution],
    observation: object,
) -> np.ndarray:
    if not predictions:
        raise ValueError("predictions must not be empty")
    normalized_prior = _normalize_prior(prior, len(predictions))
    log_weights = np.asarray(
        [
            math.log(max(float(probability), 1e-300)) + distribution.log_prob(observation)
            for probability, distribution in zip(normalized_prior, predictions)
        ],
        dtype=float,
    )
    maximum = float(np.max(log_weights))
    if not math.isfinite(maximum):
        raise ValueError("observation has zero likelihood under every hypothesis")
    weights = np.exp(log_weights - maximum)
    return weights / weights.sum()


def expected_information_gain(
    prior: np.ndarray | Sequence[float],
    predictions: Sequence[PredictiveDistribution],
) -> float:
    if not predictions:
        raise ValueError("predictions must not be empty")
    normalized_prior = _normalize_prior(prior, len(predictions))
    prior_entropy = entropy_bits(normalized_prior)
    expected_entropy = 0.0
    for hypothesis_probability, distribution in zip(normalized_prior, predictions):
        if hypothesis_probability <= 0.0:
            continue
        for observation, conditional_weight in distribution.quadrature():
            if conditional_weight <= 0.0:
                continue
            posterior = posterior_update(normalized_prior, predictions, observation)
            expected_entropy += (
                float(hypothesis_probability)
                * float(conditional_weight)
                * entropy_bits(posterior)
            )
    gain = prior_entropy - expected_entropy
    return max(0.0, float(gain))


def choose_intervention(
    hypotheses: Sequence[H],
    interventions: Sequence[I],
    predict: Callable[[H, I], PredictiveDistribution],
    prior: np.ndarray | Sequence[float],
    *,
    used: set[I] | None = None,
) -> tuple[I, float]:
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    normalized_prior = _normalize_prior(prior, len(hypotheses))
    used_set = set() if used is None else set(used)
    available = sorted((item for item in interventions if item not in used_set), key=repr)
    if not available:
        raise ValueError("no unused interventions remain")

    best_intervention = available[0]
    best_gain = -1.0
    for intervention in available:
        predictions = [predict(hypothesis, intervention) for hypothesis in hypotheses]
        gain = expected_information_gain(normalized_prior, predictions)
        if gain > best_gain + 1e-12:
            best_intervention = intervention
            best_gain = gain
    return best_intervention, float(best_gain)


def run_identification(
    hypotheses: Sequence[H],
    interventions: Sequence[I],
    predict: Callable[[H, I], PredictiveDistribution],
    observe: Callable[[I], object],
    *,
    budget: int,
    confidence: float,
    policy: str,
    rng: np.random.Generator,
    prior: np.ndarray | Sequence[float] | None = None,
) -> IdentificationResult[H, I]:
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if not interventions:
        raise ValueError("interventions must not be empty")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must lie in (0, 1]")
    if policy not in {"active", "random"}:
        raise ValueError("policy must be 'active' or 'random'")

    posterior = (
        np.full(len(hypotheses), 1.0 / len(hypotheses), dtype=float)
        if prior is None
        else _normalize_prior(prior, len(hypotheses))
    )
    used: set[I] = set()
    steps: list[IdentificationStep[I]] = []

    while len(steps) < budget and len(used) < len(interventions):
        if float(np.max(posterior)) >= confidence:
            break
        available = sorted((item for item in interventions if item not in used), key=repr)
        if policy == "active":
            intervention, information_gain = choose_intervention(
                hypotheses, interventions, predict, posterior, used=used
            )
        else:
            intervention = available[int(rng.integers(len(available)))]
            predictions_for_gain = [
                predict(hypothesis, intervention) for hypothesis in hypotheses
            ]
            information_gain = expected_information_gain(posterior, predictions_for_gain)

        before = posterior.copy()
        predictions = [predict(hypothesis, intervention) for hypothesis in hypotheses]
        observation = observe(intervention)
        posterior = posterior_update(before, predictions, observation)
        used.add(intervention)
        steps.append(
            IdentificationStep(
                intervention=intervention,
                information_gain=float(information_gain),
                observation=observation,
                prior=before,
                posterior=posterior.copy(),
                confidence=float(np.max(posterior)),
            )
        )

    map_index = int(np.argmax(posterior))
    return IdentificationResult(
        map_hypothesis=hypotheses[map_index],
        posterior=posterior.copy(),
        steps=tuple(steps),
        reached_confidence=float(np.max(posterior)) >= confidence,
        probes_used=len(steps),
    )
