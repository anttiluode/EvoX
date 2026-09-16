from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evolution import evolve
from .program import Node, mutate_tree, random_tree


@dataclass(frozen=True)
class TransferResult:
    condition: str
    total_evaluations: int
    evaluations_to_exact: int | None
    success: bool
    best_mse: float


def _seed_population(source_programs: list[Node] | tuple[Node, ...], *, seed: int, population_size: int, max_depth: int) -> list[Node]:
    if not source_programs:
        raise ValueError("source_programs must not be empty for seeded transfer")
    rng = np.random.default_rng(seed)
    trees: list[Node] = []
    for _ in range(population_size):
        parent = source_programs[int(rng.integers(len(source_programs)))]
        kind = int(rng.integers(4))
        if kind == 0:
            child = mutate_tree(parent, rng, max_depth)
        else:
            op = ("add", "sub", "mul")[kind - 1]
            # Structure-preserving mutation: keep the discovered computation
            # intact and compose one small terminal-scale operation around it.
            child = Node(op, (parent, random_tree(rng, 0)))
        trees.append(child)
    return trees


def run_transfer_condition(
    condition: str,
    *,
    source_programs: list[Node] | tuple[Node, ...] | None,
    examples,
    seed: int,
    population_size: int,
    generations: int,
    elite_count: int,
    max_depth: int,
    viable_mse: float,
) -> TransferResult:
    initial = None
    if source_programs is not None:
        initial = _seed_population(source_programs, seed=seed ^ 0xA5A5A5A5, population_size=population_size, max_depth=max_depth)
    result = evolve(
        examples,
        seed=seed,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        max_depth=max_depth,
        viable_mse=viable_mse,
        initial_trees=initial,
    )
    exact_index = next((index + 1 for index, ind in enumerate(result.ledger) if ind.mse <= viable_mse), None)
    best_mse = min(ind.mse for ind in result.ledger)
    return TransferResult(
        condition=condition,
        total_evaluations=result.evaluations,
        evaluations_to_exact=exact_index,
        success=exact_index is not None,
        best_mse=float(best_mse),
    )
