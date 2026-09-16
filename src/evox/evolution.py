from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .program import Node, evaluate, mutate_tree, node_count, random_tree, syntax_features, to_source

Example = tuple[tuple[int, int, int], float]


@dataclass(frozen=True)
class Individual:
    id: int
    generation: int
    parent_id: int | None
    root_id: int
    tree: Node
    mse: float
    nodes: int


@dataclass(frozen=True)
class EvolutionResult:
    ledger: tuple[Individual, ...]
    final_population: tuple[Individual, ...]
    viable: tuple[Individual, ...]
    evaluations: int
    archive: tuple[Individual, ...] = ()


def program_mse(tree: Node, examples: list[Example] | tuple[Example, ...]) -> float:
    if not examples:
        raise ValueError("examples must not be empty")
    errors = [(evaluate(tree, x) - target) ** 2 for x, target in examples]
    return float(np.mean(errors))


def _make_individual(
    ident: int,
    generation: int,
    parent_id: int | None,
    root_id: int,
    tree: Node,
    examples: list[Example] | tuple[Example, ...],
) -> Individual:
    return Individual(
        id=ident,
        generation=generation,
        parent_id=parent_id,
        root_id=root_id,
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


def _select_structural_novelty(
    population: list[Individual], elite_count: int, novelty_fraction: float = 0.5
) -> list[Individual]:
    ranked = sorted(population, key=_rank_key)
    baseline_count = max(1, elite_count - int(round(elite_count * novelty_fraction)))
    selected = _select_elites(population, baseline_count)
    selected_sources = {to_source(ind.tree) for ind in selected}

    best_mse = ranked[0].mse
    candidates = [ind for ind in ranked if abs(ind.mse - best_mse) <= 1e-15]
    unique_candidates: list[Individual] = []
    seen: set[str] = set()
    for ind in candidates:
        source = to_source(ind.tree)
        if source in seen:
            continue
        seen.add(source)
        unique_candidates.append(ind)

    if unique_candidates:
        matrix = np.stack([syntax_features(ind.tree) for ind in unique_candidates])
        scale = np.std(matrix, axis=0)
        scale[scale == 0.0] = 1.0
        normalized = (matrix - np.mean(matrix, axis=0)) / scale
        index_by_id = {ind.id: i for i, ind in enumerate(unique_candidates)}

        while len(selected) < elite_count:
            available = [ind for ind in unique_candidates if to_source(ind.tree) not in selected_sources]
            if not available:
                break
            selected_vectors = [normalized[index_by_id[ind.id]] for ind in selected if ind.id in index_by_id]
            if not selected_vectors:
                choice = available[0]
            else:
                best_choice = None
                best_distance = -1.0
                for ind in available:
                    vector = normalized[index_by_id[ind.id]]
                    distance = min(float(np.linalg.norm(vector - other)) for other in selected_vectors)
                    if distance > best_distance + 1e-12:
                        best_choice = ind
                        best_distance = distance
                    elif (
                        abs(distance - best_distance) <= 1e-12
                        and best_choice is not None
                        and _rank_key(ind) < _rank_key(best_choice)
                    ):
                        best_choice = ind
                choice = best_choice if best_choice is not None else available[0]
            selected.append(choice)
            selected_sources.add(to_source(choice.tree))

    if len(selected) < elite_count:
        for ind in _select_elites(population, elite_count):
            if len(selected) == elite_count:
                break
            if ind.id not in {item.id for item in selected}:
                selected.append(ind)
    return selected[:elite_count]


def _select_lineage_niching(population: list[Individual], elite_count: int) -> list[Individual]:
    ranked = sorted(population, key=_rank_key)
    selected: list[Individual] = []
    seen_roots: set[int] = set()
    for ind in ranked:
        if ind.root_id in seen_roots:
            continue
        seen_roots.add(ind.root_id)
        selected.append(ind)
        if len(selected) == elite_count:
            return selected
    for ind in _select_elites(population, elite_count):
        if len(selected) == elite_count:
            break
        if ind.id not in {item.id for item in selected}:
            selected.append(ind)
    return selected[:elite_count]


def _select_elites_for_strategy(
    population: list[Individual], elite_count: int, strategy: str, novelty_fraction: float = 0.5
) -> list[Individual]:
    if strategy in {"baseline", "neutral_archive"}:
        return _select_elites(population, elite_count)
    if strategy == "structural_novelty":
        return _select_structural_novelty(population, elite_count, novelty_fraction)
    if strategy == "lineage_niching":
        return _select_lineage_niching(population, elite_count)
    raise ValueError(f"unknown strategy: {strategy}")


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
    strategy: str = "baseline",
    archive_capacity: int = 64,
    archive_fraction: float = 0.25,
    novelty_fraction: float = 0.5,
) -> EvolutionResult:
    if population_size <= 0 or generations <= 0:
        raise ValueError("population_size and generations must be positive")
    if not 1 <= elite_count <= population_size:
        raise ValueError("elite_count must be between 1 and population_size")
    if strategy not in {"baseline", "structural_novelty", "lineage_niching", "neutral_archive"}:
        raise ValueError(f"unknown strategy: {strategy}")
    if archive_capacity <= 0:
        raise ValueError("archive_capacity must be positive")
    if not 0.0 <= archive_fraction <= 1.0:
        raise ValueError("archive_fraction must be between 0 and 1")
    if not 0.0 <= novelty_fraction <= 1.0:
        raise ValueError("novelty_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    ledger: list[Individual] = []
    ident = 0

    trees: list[Node] = list(initial_trees or ())[:population_size]
    while len(trees) < population_size:
        trees.append(random_tree(rng, max_depth))

    population: list[Individual] = []
    archive_by_source: dict[str, Individual] = {}
    archive_cursor = 0
    for tree in trees:
        ind = _make_individual(ident, 0, None, ident, tree, examples)
        ident += 1
        population.append(ind)
        ledger.append(ind)

    def update_archive(current: list[Individual]) -> None:
        if strategy != "neutral_archive":
            return
        for item in sorted(current, key=_rank_key):
            if item.mse > viable_mse:
                continue
            source = to_source(item.tree)
            if source in archive_by_source:
                continue
            if len(archive_by_source) >= archive_capacity:
                break
            archive_by_source[source] = item

    update_archive(population)

    for generation in range(1, generations):
        if strategy == "neutral_archive":
            archive_slots = (
                min(elite_count, max(1, int(round(elite_count * archive_fraction))))
                if archive_by_source
                else 0
            )
            current_slots = elite_count - archive_slots
            elites = _select_elites_for_strategy(population, current_slots, "baseline") if current_slots else []
            selected_sources = {to_source(item.tree) for item in elites}
            archived_items = list(archive_by_source.values())
            attempts = 0
            while len(elites) < elite_count and archived_items and attempts < len(archived_items) * 2:
                item = archived_items[archive_cursor % len(archived_items)]
                archive_cursor += 1
                attempts += 1
                source = to_source(item.tree)
                if source in selected_sources:
                    continue
                elites.append(item)
                selected_sources.add(source)
            if len(elites) < elite_count:
                for item in _select_elites(population, elite_count):
                    if len(elites) == elite_count:
                        break
                    if item.id not in {existing.id for existing in elites}:
                        elites.append(item)
        else:
            elites = _select_elites_for_strategy(population, elite_count, strategy, novelty_fraction)
        next_population: list[Individual] = []

        # Re-evaluate elite copies so every generation has exactly population_size evaluations.
        for elite in elites:
            clone = _make_individual(ident, generation, elite.id, elite.root_id, elite.tree, examples)
            ident += 1
            next_population.append(clone)
            ledger.append(clone)

        while len(next_population) < population_size:
            parent = elites[int(rng.integers(len(elites)))]
            tree = mutate_tree(parent.tree, rng, max_depth)
            child = _make_individual(ident, generation, parent.id, parent.root_id, tree, examples)
            ident += 1
            next_population.append(child)
            ledger.append(child)
        population = next_population
        update_archive(population)

    viable = tuple(sorted((ind for ind in population if ind.mse <= viable_mse), key=_rank_key))
    return EvolutionResult(
        ledger=tuple(ledger),
        final_population=tuple(population),
        viable=viable,
        evaluations=population_size * generations,
        archive=tuple(archive_by_source.values()),
    )
