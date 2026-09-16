from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .program import Node, evaluate, mutate_tree, node_count, random_tree, to_source

Example = tuple[tuple[int, int, int], float]


@dataclass(frozen=True)
class Individual:
    id: int
    generation: int
    parent_id: int | None
    tree: Node
    mse: float
    nodes: int


@dataclass(frozen=True)
class EvolutionResult:
    ledger: tuple[Individual, ...]
    final_population: tuple[Individual, ...]
    viable: tuple[Individual, ...]
    evaluations: int


def program_mse(tree: Node, examples: list[Example] | tuple[Example, ...]) -> float:
    if not examples:
        raise ValueError("examples must not be empty")
    errors = [(evaluate(tree, x) - target) ** 2 for x, target in examples]
    return float(np.mean(errors))


def _make_individual(
    ident: int,
    generation: int,
    parent_id: int | None,
    tree: Node,
    examples: list[Example] | tuple[Example, ...],
) -> Individual:
    return Individual(
        id=ident,
        generation=generation,
        parent_id=parent_id,
        tree=tree,
        mse=program_mse(tree, examples),
        nodes=node_count(tree),
    )


def _rank_key(ind: Individual) -> tuple[float, int, str, int]:
    return (ind.mse, ind.nodes, to_source(ind.tree), ind.id)


def _select_elites(population: list[Individual], elite_count: int) -> list[Individual]:
    ranked = sorted(population, key=_rank_key)
    elites: list[Individual] = []
    seen: set[str] = set()
    for ind in ranked:
        source = to_source(ind.tree)
        if source in seen:
            continue
        seen.add(source)
        elites.append(ind)
        if len(elites) == elite_count:
            return elites
    for ind in ranked:
        if len(elites) == elite_count:
            break
        elites.append(ind)
    return elites


def evolve(
    examples: list[Example] | tuple[Example, ...],
    *,
    seed: int,
    population_size: int,
    generations: int,
    elite_count: int,
    max_depth: int,
    viable_mse: float,
    initial_trees: list[Node] | tuple[Node, ...] | None = None,
) -> EvolutionResult:
    if population_size <= 0 or generations <= 0:
        raise ValueError("population_size and generations must be positive")
    if not 1 <= elite_count <= population_size:
        raise ValueError("elite_count must be between 1 and population_size")

    rng = np.random.default_rng(seed)
    ledger: list[Individual] = []
    ident = 0

    trees: list[Node] = list(initial_trees or ())[:population_size]
    while len(trees) < population_size:
        trees.append(random_tree(rng, max_depth))

    population: list[Individual] = []
    for tree in trees:
        ind = _make_individual(ident, 0, None, tree, examples)
        ident += 1
        population.append(ind)
        ledger.append(ind)

    for generation in range(1, generations):
        elites = _select_elites(population, elite_count)
        next_population: list[Individual] = []

        # Re-evaluate elite copies so every generation has exactly population_size evaluations.
        for elite in elites:
            clone = _make_individual(ident, generation, elite.id, elite.tree, examples)
            ident += 1
            next_population.append(clone)
            ledger.append(clone)

        while len(next_population) < population_size:
            parent = elites[int(rng.integers(len(elites)))]
            tree = mutate_tree(parent.tree, rng, max_depth)
            child = _make_individual(ident, generation, parent.id, tree, examples)
            ident += 1
            next_population.append(child)
            ledger.append(child)
        population = next_population

    viable = tuple(sorted((ind for ind in population if ind.mse <= viable_mse), key=_rank_key))
    return EvolutionResult(
        ledger=tuple(ledger),
        final_population=tuple(population),
        viable=viable,
        evaluations=population_size * generations,
    )
