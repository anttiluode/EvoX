from __future__ import annotations

import numpy as np

from evox.active import GaussianDistribution
from evox.domains.common import AdaptationResult, DomainCase

HYPOTHESES = ("c0", "c1", "c2", "c3")
COEFFICIENTS = {
    "c0": {"A": 1.0, "B": 0.4, "C": -0.5},
    "c1": {"A": 0.5, "B": -1.0, "C": 0.2},
    "c2": {"A": 0.2, "B": 0.5, "C": 1.0},
    "c3": {"A": -1.0, "B": 0.2, "C": 0.5},
}
PASSIVE = ("none", 0.0)
INTERVENTIONS = (
    PASSIVE,
    ("A", 1.0),
    ("B", 1.0),
    ("C", 1.0),
    ("A", -0.5),
    ("B", -0.5),
    ("C", -0.5),
)
OBSERVATION_SIGMA = 0.12
BASELINE_OUTPUT = 1.0
DISTURBED_OUTPUT = 0.2
TARGET_OUTPUT = 1.0
TARGET_TOLERANCE = 0.05
COMPENSATIONS = tuple(
    (node, amount)
    for node in ("A", "B", "C")
    for amount in (-1.6, -0.8, 0.8, 1.6)
)


def _intervention_mean(hypothesis: str, intervention: tuple[str, float]) -> float:
    node, amount = intervention
    if node == "none":
        return BASELINE_OUTPUT
    return float(BASELINE_OUTPUT + COEFFICIENTS[hypothesis][node] * float(amount))


def _predict(hypothesis: str, intervention: tuple[str, float]):
    return GaussianDistribution(_intervention_mean(hypothesis, intervention), OBSERVATION_SIGMA)


def _observe(hypothesis: str, intervention: tuple[str, float], rng: np.random.Generator):
    mean = _intervention_mean(hypothesis, intervention)
    return float(rng.normal(mean, OBSERVATION_SIGMA))


def _compensated_output(hypothesis: str, intervention: tuple[str, float]) -> float:
    node, amount = intervention
    return float(DISTURBED_OUTPUT + COEFFICIENTS[hypothesis][node] * float(amount))


def _rank_compensations(selected: str | None) -> tuple[tuple[str, float], ...]:
    if selected is None:
        return tuple(sorted(COMPENSATIONS, key=repr))
    return tuple(
        sorted(
            COMPENSATIONS,
            key=lambda item: (
                abs(_compensated_output(selected, item) - TARGET_OUTPUT),
                abs(item[1]),
                repr(item),
            ),
        )
    )


def _try_compensations(
    order: tuple[tuple[str, float], ...], truth: str, budget: int
) -> AdaptationResult:
    executions = 0
    best_error = float("inf")
    for intervention in order:
        if executions >= budget:
            break
        executions += 1
        error = abs(_compensated_output(truth, intervention) - TARGET_OUTPUT)
        best_error = min(best_error, error)
        if error <= TARGET_TOLERANCE:
            return AdaptationResult(True, executions, float(error), f"compensation:{intervention}")
    return AdaptationResult(False, executions, float(best_error), "output not restored")


def _adapt(selected: str, truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_compensations(_rank_compensations(selected), truth, budget)


def _restart(truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_compensations(_rank_compensations(None), truth, budget)


def build_case() -> DomainCase[str, tuple[str, float]]:
    return DomainCase(
        name="causal_circuit",
        hypotheses=HYPOTHESES,
        interventions=INTERVENTIONS,
        passive_intervention=PASSIVE,
        predict=_predict,
        observe=_observe,
        adapt=_adapt,
        restart=_restart,
        probe_budget=2,
        adapt_budget=len(COMPENSATIONS),
    )
