from __future__ import annotations

import numpy as np

from evox.active import CategoricalDistribution
from evox.domains.common import AdaptationResult, DomainCase

HYPOTHESES = ("xor_post", "xor_pre", "track_post", "track_pre")
PASSIVE = (0,)
INTERVENTIONS = (
    PASSIVE,
    (1,),
    (1, 0),
    (1, 1),
    (1, 0, 1),
    (0, 1, 0),
)
CONTROL_WORDS = (
    (0,),
    (1,),
    (0, 1),
    (1, 0),
    (1, 1),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def _step(hypothesis: str, state: int, bit: int) -> tuple[int, int]:
    if hypothesis == "xor_post":
        next_state = state ^ bit
        return next_state, next_state
    if hypothesis == "xor_pre":
        next_state = state ^ bit
        return next_state, state
    if hypothesis == "track_post":
        next_state = bit
        return next_state, next_state
    if hypothesis == "track_pre":
        next_state = bit
        return next_state, state
    raise KeyError(hypothesis)


def _run(hypothesis: str, word: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    state = 0
    outputs: list[int] = []
    for bit in word:
        state, output = _step(hypothesis, state, int(bit))
        outputs.append(output)
    return tuple(outputs), state


def _predict(hypothesis: str, intervention: tuple[int, ...]):
    outputs, _ = _run(hypothesis, intervention)
    return CategoricalDistribution({outputs: 1.0})


def _observe(hypothesis: str, intervention: tuple[int, ...], rng: np.random.Generator):
    del rng
    outputs, _ = _run(hypothesis, intervention)
    return outputs


def _objective(hypothesis: str, word: tuple[int, ...]) -> bool:
    outputs, final_state = _run(hypothesis, word)
    return bool(outputs) and final_state == 1 and outputs[-1] == 1


def _rank_words(selected: str | None) -> tuple[tuple[int, ...], ...]:
    if selected is None:
        return tuple(sorted(CONTROL_WORDS, key=lambda word: (len(word), word)))
    return tuple(
        sorted(
            CONTROL_WORDS,
            key=lambda word: (0 if _objective(selected, word) else 1, len(word), word),
        )
    )


def _try_words(order: tuple[tuple[int, ...], ...], truth: str, budget: int) -> AdaptationResult:
    executions = 0
    for word in order:
        if executions >= budget:
            break
        executions += 1
        if _objective(truth, word):
            return AdaptationResult(True, executions, 0.0, f"control:{word}")
    return AdaptationResult(False, executions, 1.0, "target state/output not reached")


def _adapt(selected: str, truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_words(_rank_words(selected), truth, budget)


def _restart(truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _try_words(_rank_words(None), truth, budget)


def build_case() -> DomainCase[str, tuple[int, ...]]:
    return DomainCase(
        name="state_machine",
        hypotheses=HYPOTHESES,
        interventions=INTERVENTIONS,
        passive_intervention=PASSIVE,
        predict=_predict,
        observe=_observe,
        adapt=_adapt,
        restart=_restart,
        probe_budget=2,
        adapt_budget=len(CONTROL_WORDS),
    )
