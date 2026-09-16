from __future__ import annotations

import numpy as np

from evox.active import CategoricalDistribution
from evox.domains.common import AdaptationResult, DomainCase

HYPOTHESES = ("reverse", "rotate_left", "rotate_right", "swap_pairs")
PASSIVE = (0, 0, 0, 0)
INTERVENTIONS = (
    PASSIVE,
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (0, 1, 2, 3),
)
FOLLOWUP_PAYLOAD = (9, 4, 7, 2, 5, 1)


def _transform(hypothesis: str, values: tuple[int, ...]) -> tuple[int, ...]:
    if hypothesis == "reverse":
        return tuple(reversed(values))
    if hypothesis == "rotate_left":
        return values[1:] + values[:1]
    if hypothesis == "rotate_right":
        return values[-1:] + values[:-1]
    if hypothesis == "swap_pairs":
        result = list(values)
        for index in range(0, len(result) - 1, 2):
            result[index], result[index + 1] = result[index + 1], result[index]
        return tuple(result)
    raise KeyError(hypothesis)


def _predict(hypothesis: str, intervention: tuple[int, ...]):
    return CategoricalDistribution({_transform(hypothesis, intervention): 1.0})


def _observe(hypothesis: str, intervention: tuple[int, ...], rng: np.random.Generator):
    del rng
    return _transform(hypothesis, intervention)


def _candidate_order(selected: str | None) -> tuple[str, ...]:
    if selected is None:
        return HYPOTHESES
    return (selected,) + tuple(item for item in HYPOTHESES if item != selected)


def _validate(order: tuple[str, ...], truth: str, budget: int) -> AdaptationResult:
    expected = _transform(truth, FOLLOWUP_PAYLOAD)
    executions = 0
    for candidate in order:
        if executions >= budget:
            break
        executions += 1
        output = _transform(candidate, FOLLOWUP_PAYLOAD)
        if output == expected:
            return AdaptationResult(True, executions, 0.0, f"transform:{candidate}")
    return AdaptationResult(False, executions, 1.0, "no transformation validated")


def _adapt(selected: str, truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _validate(_candidate_order(selected), truth, budget)


def _restart(truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _validate(_candidate_order(None), truth, budget)


def build_case() -> DomainCase[str, tuple[int, ...]]:
    return DomainCase(
        name="program_transform",
        hypotheses=HYPOTHESES,
        interventions=INTERVENTIONS,
        passive_intervention=PASSIVE,
        predict=_predict,
        observe=_observe,
        adapt=_adapt,
        restart=_restart,
        probe_budget=2,
        adapt_budget=len(HYPOTHESES),
    )
