from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar

import numpy as np

from evox.active import PredictiveDistribution

H = TypeVar("H", bound=Hashable)
I = TypeVar("I", bound=Hashable)


@dataclass(frozen=True)
class AdaptationResult:
    success: bool
    executions: int
    error: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DomainCase(Generic[H, I]):
    name: str
    hypotheses: tuple[H, ...]
    interventions: tuple[I, ...]
    passive_intervention: I
    predict: Callable[[H, I], PredictiveDistribution]
    observe: Callable[[H, I, np.random.Generator], object]
    adapt: Callable[[H, H, int, np.random.Generator], AdaptationResult]
    restart: Callable[[H, int, np.random.Generator], AdaptationResult]
    probe_budget: int
    adapt_budget: int
