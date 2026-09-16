from evox.evolution import evolve
from evox.program import to_source
from evox.worlds import visible_examples


def test_evolution_records_parent_generation_and_evaluation_count():
    result = evolve(visible_examples(), seed=3, population_size=24, generations=4, elite_count=6, max_depth=3, viable_mse=0.0)
    assert result.evaluations == 24 * 4
    assert result.ledger
    assert all(ind.generation >= 0 for ind in result.ledger)
    assert any(ind.parent_id is not None for ind in result.ledger if ind.generation > 0)


def test_viable_programs_are_selected_only_by_visible_fitness():
    result = evolve(visible_examples(), seed=9, population_size=48, generations=8, elite_count=12, max_depth=3, viable_mse=1e-12)
    assert all(ind.mse <= 1e-12 for ind in result.viable)


def test_evolution_replays_deterministically():
    kwargs = dict(seed=11, population_size=32, generations=5, elite_count=8, max_depth=3, viable_mse=1e-12)
    a = evolve(visible_examples(), **kwargs)
    b = evolve(visible_examples(), **kwargs)
    assert [(to_source(i.tree), i.mse) for i in a.final_population] == [(to_source(i.tree), i.mse) for i in b.final_population]
