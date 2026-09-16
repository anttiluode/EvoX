from __future__ import annotations

import numpy as np

from evox.active import CategoricalDistribution
from evox.domains.common import AdaptationResult, DomainCase

HYPOTHESES = ("cache_key", "parser_mode", "auth_scope", "retry_state")
PASSIVE = "default_request"
INTERVENTIONS = (
    PASSIVE,
    "split_cache_parser_vs_auth_retry",
    "split_cache_auth_vs_parser_retry",
    "cache_probe",
    "parser_probe",
    "auth_probe",
    "retry_probe",
)


def _diagnostic_outcome(hypothesis: str, intervention: str) -> str:
    if intervention == PASSIVE:
        return "pass"
    if intervention == "split_cache_parser_vs_auth_retry":
        return "left-signature" if hypothesis in {"cache_key", "parser_mode"} else "right-signature"
    if intervention == "split_cache_auth_vs_parser_retry":
        return "left-signature" if hypothesis in {"cache_key", "auth_scope"} else "right-signature"
    if intervention.endswith("_probe"):
        named = intervention.removesuffix("_probe")
        aliases = {
            "cache": "cache_key",
            "parser": "parser_mode",
            "auth": "auth_scope",
            "retry": "retry_state",
        }
        return f"{named}-failure" if aliases[named] == hypothesis else "pass"
    raise KeyError(intervention)


def _predict(hypothesis: str, intervention: str):
    return CategoricalDistribution({_diagnostic_outcome(hypothesis, intervention): 1.0})


def _observe(hypothesis: str, intervention: str, rng: np.random.Generator):
    del rng
    return _diagnostic_outcome(hypothesis, intervention)


def _patch_order(selected: str | None) -> tuple[str, ...]:
    if selected is None:
        return HYPOTHESES
    return (selected,) + tuple(item for item in HYPOTHESES if item != selected)


def _validate_patches(order: tuple[str, ...], truth: str, budget: int) -> AdaptationResult:
    executions = 0
    for patch in order:
        if executions >= budget:
            break
        executions += 1
        if patch == truth:
            return AdaptationResult(True, executions, 0.0, f"validated patch:{patch}")
    return AdaptationResult(False, executions, 1.0, "no patch validated inside budget")


def _adapt(selected: str, truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _validate_patches(_patch_order(selected), truth, budget)


def _restart(truth: str, budget: int, rng: np.random.Generator) -> AdaptationResult:
    del rng
    return _validate_patches(_patch_order(None), truth, budget)


def build_case() -> DomainCase[str, str]:
    return DomainCase(
        name="regression_fault",
        hypotheses=HYPOTHESES,
        interventions=INTERVENTIONS,
        passive_intervention=PASSIVE,
        predict=_predict,
        observe=_observe,
        adapt=_adapt,
        restart=_restart,
        probe_budget=2,
        adapt_budget=4,
    )
