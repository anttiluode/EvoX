import numpy as np

from evox.evolution import Individual, _select_elites_for_strategy, evolve
from evox.program import Node, node_count, syntax_features, to_source, tree_depth
from evox.worlds import visible_examples
from experiments.run_v1_diversity import V1Config, classify_strategy, run_strategy_seed


def test_syntax_features_distinguish_direct_and_relational_programs():
    direct = Node("x0")
    relational = Node("mul", (Node("x1"), Node("x2")))

    assert tree_depth(direct) == 1
    assert tree_depth(relational) == 2
    np.testing.assert_array_equal(
        syntax_features(direct),
        np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    )
    np.testing.assert_array_equal(
        syntax_features(relational),
        np.asarray([3, 2, 0, 1, 1, 0, 0, 0, 0, 1], dtype=float),
    )


def test_root_lineage_is_inherited_across_generations():
    result = evolve(
        visible_examples(),
        seed=31,
        population_size=20,
        generations=4,
        elite_count=5,
        max_depth=3,
        viable_mse=1e-12,
    )
    by_id = {ind.id: ind for ind in result.ledger}
    for ind in result.ledger:
        if ind.generation == 0:
            assert ind.root_id == ind.id
        else:
            assert ind.parent_id is not None
            assert ind.root_id == by_id[ind.parent_id].root_id


def test_explicit_baseline_matches_default_evolution():
    kwargs = dict(
        seed=37,
        population_size=32,
        generations=5,
        elite_count=8,
        max_depth=3,
        viable_mse=1e-12,
    )
    default = evolve(visible_examples(), **kwargs)
    baseline = evolve(visible_examples(), strategy="baseline", **kwargs)
    assert [(to_source(i.tree), i.mse, i.parent_id) for i in default.ledger] == [
        (to_source(i.tree), i.mse, i.parent_id) for i in baseline.ledger
    ]


def _individual(ident, tree, *, root_id=None, mse=0.0):
    return Individual(
        id=ident,
        generation=0,
        parent_id=None,
        root_id=ident if root_id is None else root_id,
        tree=tree,
        mse=mse,
        nodes=node_count(tree),
    )


def test_structural_novelty_spends_elite_slot_on_syntax_distance():
    direct = Node("x0")
    direct_add_zero = Node("add", (Node("x0"), Node("const0")))
    direct_sub_zero = Node("sub", (Node("x0"), Node("const0")))
    relational = Node("mul", (Node("x1"), Node("x2")))
    population = [
        _individual(0, direct),
        _individual(1, direct_add_zero),
        _individual(2, direct_sub_zero),
        _individual(3, relational),
    ]

    selected = _select_elites_for_strategy(population, 2, "structural_novelty")
    sources = {to_source(ind.tree) for ind in selected}
    assert "x0" in sources
    assert "(x1*x2)" in sources


def test_lineage_niching_selects_distinct_roots_before_duplicates():
    population = [
        _individual(0, Node("x0"), root_id=10, mse=0.0),
        _individual(1, Node("add", (Node("x0"), Node("const0"))), root_id=10, mse=0.0),
        _individual(2, Node("mul", (Node("x1"), Node("x2"))), root_id=20, mse=0.0),
        _individual(3, Node("sub", (Node("x0"), Node("const0"))), root_id=30, mse=0.0),
    ]
    selected = _select_elites_for_strategy(population, 3, "lineage_niching")
    assert {ind.root_id for ind in selected} == {10, 20, 30}


def test_neutral_archive_contains_only_visible_viable_programs_and_keeps_budget_equal():
    initial = [
        Node("x0"),
        Node("mul", (Node("x1"), Node("x2"))),
        Node("const2"),
        Node("const_m2"),
    ]
    common = dict(
        examples=visible_examples(),
        seed=41,
        population_size=24,
        generations=4,
        elite_count=6,
        max_depth=3,
        viable_mse=1e-12,
        initial_trees=initial,
    )
    baseline = evolve(strategy="baseline", **common)
    archived = evolve(
        strategy="neutral_archive",
        archive_capacity=16,
        archive_fraction=0.25,
        **common,
    )
    assert archived.evaluations == baseline.evaluations == 24 * 4
    assert archived.archive
    assert all(ind.mse <= 1e-12 for ind in archived.archive)
    assert len({to_source(ind.tree) for ind in archived.archive}) == len(archived.archive)


def test_v1_smoke_reports_minority_mass_and_equal_budget():
    config = V1Config(population_size=32, generations=4, elite_count=8, canonical_seed_count=1)
    rows = [
        run_strategy_seed(strategy, 501, config)
        for strategy in ("baseline", "structural_novelty", "lineage_niching", "neutral_archive")
    ]
    assert {row["strategy"] for row in rows} == {
        "baseline",
        "structural_novelty",
        "lineage_niching",
        "neutral_archive",
    }
    assert {row["evolution_evaluations"] for row in rows} == {32 * 4}
    for row in rows:
        if row["status"] == "OK":
            assert 0.0 < row["minority_mode_fraction"] <= 0.5


def test_v1_classification_distinguishes_preserved_from_thin_modes():
    baseline = {"median_best_visible_mse": 0.0}
    good = {
        "valid_seed_fraction": 0.9,
        "mean_minority_mode_fraction": 0.2,
        "mean_mode_purity": 1.0,
        "median_best_visible_mse": 0.0,
        "active_accuracy": 1.0,
        "random_accuracy": 1.0,
        "active_mean_censored_probes": 1.0,
        "random_mean_censored_probes": 2.0,
        "transfer": {
            "correct": {"mean_censored_evaluations": 20.0, "success_rate": 1.0},
            "restart": {"mean_censored_evaluations": 100.0, "success_rate": 0.8},
        },
    }
    assert classify_strategy(good, baseline) == "PRESERVES_MODES"
    thin = dict(good, mean_minority_mode_fraction=0.1)
    assert classify_strategy(thin, baseline) == "SURVIVES_BUT_THIN"


def test_structural_novelty_selection_is_stable_under_population_order():
    trees = [
        Node("x0"),
        Node("add", (Node("x0"), Node("const0"))),
        Node("sub", (Node("x0"), Node("const0"))),
        Node("mul", (Node("x1"), Node("x2"))),
        Node("add", (Node("mul", (Node("x1"), Node("x2"))), Node("const0"))),
    ]
    population = [_individual(i, tree) for i, tree in enumerate(trees)]
    forward = _select_elites_for_strategy(population, 3, "structural_novelty", novelty_fraction=0.75)
    reverse = _select_elites_for_strategy(
        list(reversed(population)), 3, "structural_novelty", novelty_fraction=0.75
    )
    assert [ind.id for ind in forward] == [ind.id for ind in reverse]
