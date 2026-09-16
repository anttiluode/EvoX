from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from .program import Node, evaluate

Input = tuple[int, int, int]


def _sample_without_replacement(
    items: Sequence[Input], *, seed: int, length: int
) -> list[Input]:
    if length <= 0:
        raise ValueError("length must be positive")
    if length > len(items):
        raise ValueError("length exceeds available unique descriptor inputs")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=length, replace=False)
    return [items[int(index)] for index in indices]


def generic_unlabeled_tape(*, seed: int, length: int = 32) -> list[Input]:
    """Sample inputs from a fixed cube without consulting any target mechanism."""
    values = (-3, -2, -1, 0, 1, 2, 3)
    cube = list(product(values, repeat=3))
    return _sample_without_replacement(cube, seed=seed, length=length)


def in_manifold_unlabeled_tape(*, seed: int, length: int = 32) -> list[Input]:
    """Sample unlabeled inputs from the visible-task manifold x0=x1*x2."""
    values = (-3, -2, -1, 0, 1, 2, 3)
    manifold = [(x1 * x2, x1, x2) for x1, x2 in product(values, repeat=2)]
    return _sample_without_replacement(manifold, seed=seed, length=length)


def behavior_matrix(programs: Sequence[Node], inputs: Sequence[Input]) -> np.ndarray:
    """Return program outputs on unlabeled inputs; no target vector is accepted."""
    if not programs:
        return np.empty((0, len(inputs)), dtype=float)
    return np.asarray(
        [[evaluate(program, x) for x in inputs] for program in programs],
        dtype=float,
    )
