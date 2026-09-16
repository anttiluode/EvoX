from __future__ import annotations

from itertools import product

import numpy as np

from evox.active import GaussianDistribution
from evox.domains.common import AdaptationResult, DomainCase

HYPOTHESES = ("d0", "d1", "d2", "d3")
PARAMS = {
    "d0": (0.5, 1.0),
    "d1": (0.8, 0.7),
    "d2": (-0.5, 1.0),
    "d3": (0.2, -1.0),
}
PASSIVE = (0.0, 0.0)
INTERVENTIONS = (
    PASSIVE,
    (1.0,),
    (-1.0,),
    (1.0, 1.0),
    (1.0, -1.0),
    (0.0, 1.0),
)
OBSERVATION_SIGMA = 0.15
TARGET = 1.2
TARGET_TOLERANCE = 0.08
CONTROL_LIBRARY = tuple(
    sequence
    for length in (1, 2, 3)
    for sequence in product((-1.0, 0.0, 1.0), repeat=length)
    if any(value != 0.0 for value in sequence)
)


def _final_state(hypothesis: str, controls: tuple[float, ...]) -> float:
    a, b = PARAMS[hypothesis]
    state = 0.0
    for control in controls:
        state = a * state + b * float(control)
    return float(state)


def _predict(hypothesis: str, intervention: tuple[float, ...]):
    return GaussianDistribution(_final_state(hypothesis, intervention), OBSERVATION_SIGMA)


def _observe(hypothesis: str, intervention: tuple[float, ...], rng: np.random.Generator):
    mean = _final_state(hypothesis, intervention)
    return float(rng.normal(mean, OBSERVATION_SIGMA))


def _rank_controls(selected: str | None) -> tuple[tuple[float, ...], ...]:
    if selected is None:
        return tuple(sorted(CONTROL_LIBRARY, key=lambda seq: (len(seq), seq)))
    return tuple(
        sorted(
            CONTROL_LIBRARY,
            key=lambda seq: (
                abs(_final_state(selected, seq) - TARGET),
                len(seq),
                sum(abs(value) for value in seq),
                seq,
            ),
        )
    )


def _try_controls(order: tuple[tuple[float, ...], ...], truth: str, budget: int) -> AdaptationResult:
    executions = 0
    best_error = float("inf")
    for controls in order:
        if executions >= budget:
            break
        executions += 1
        error = abs(_final_state(truth, controls) - TARGET)
        best_error = min(best_error, error)
        if error <= TARGET_TOLERANCE:
            return AdaptationResult(True, executions, float(error), f"control:{controls}")
    return AdaptationResult(False, executions, float(best_error), "target state not reached")


def _adapt(selected: str, truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_controls(_rank_controls(selected), truth, budget)


def _restart(truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_controls(_rank_controls(None), truth, budget)


def build_case() -> DomainCase[str, tuple[float, ...]]:
    return DomainCase(
        name="dynamics",
        hypotheses=HYPOTHESES,
        interventions=INTERVENTIONS,
        passive_intervention=PASSIVE,
        predict=_predict,
        observe=_observe,
        adapt=_adapt,
        restart=_restart,
        probe_budget=2,
        adapt_budget=12,
    )
