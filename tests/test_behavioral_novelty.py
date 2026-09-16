import numpy as np

from evox.descriptors import generic_unlabeled_tape, in_manifold_unlabeled_tape
from evox.evolution import Individual, _select_elites_for_strategy, evolve
from evox.program import Node, node_count, to_source
from evox.worlds import visible_examples


def _individual(ident: int, tree: Node, *, mse: float = 0.0) -> Individual:
    return Individual(
        id=ident,
        generation=0,
        parent_id=None,
        root_id=ident,
        tree=tree,
        mse=mse,
        nodes=node_count(tree),
    )


def test_behavioral_novelty_spends_slot_on_unlabeled_functional_difference():
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
    tape = generic_unlabeled_tape(seed=271828, length=32)
    selected = _select_elites_for_strategy(
        population,
        2,
        "behavioral_novelty",
        novelty_fraction=0.5,
        descriptor_inputs=tape,
    )
    sources = {to_source(ind.tree) for ind in selected}
    assert "x0" in sources
    assert "(x1*x2)" in sources


def test_in_manifold_behavioral_novelty_has_no_reference_difference_to_exploit():
    direct = Node("x0")
    relational = Node("mul", (Node("x1"), Node("x2")))
    population = [_individual(0, direct), _individual(1, relational)]
    tape = in_manifold_unlabeled_tape(seed=314159, length=32)
    selected = _select_elites_for_strategy(
        population,
        2,
        "behavioral_novelty",
        novelty_fraction=0.75,
        descriptor_inputs=tape,
    )
    assert {to_source(ind.tree) for ind in selected} == {"x0", "(x1*x2)"}
    # Selection may retain both because both elite slots are available, but their
    # unlabeled descriptor distance itself is exactly zero on this control tape.
    outputs = np.asarray([[x[0] for x in tape], [x[1] * x[2] for x in tape]], dtype=float)
    assert np.linalg.norm(outputs[0] - outputs[1]) == 0.0


def test_behavioral_selection_is_invariant_to_common_descriptor_permutation():
    trees = [
        Node("x0"),
        Node("add", (Node("x0"), Node("const0"))),
        Node("mul", (Node("x1"), Node("x2"))),
        Node("x1"),
    ]
    population = [_individual(i, tree) for i, tree in enumerate(trees)]
    tape = generic_unlabeled_tape(seed=271828, length=32)
    permutation = np.random.default_rng(7).permutation(len(tape))
    permuted = [tape[int(i)] for i in permutation]
    first = _select_elites_for_strategy(
        population,
        3,
        "behavioral_novelty",
        novelty_fraction=0.75,
        descriptor_inputs=tape,
    )
    second = _select_elites_for_strategy(
        population,
        3,
        "behavioral_novelty",
        novelty_fraction=0.75,
        descriptor_inputs=permuted,
    )
    assert [ind.id for ind in first] == [ind.id for ind in second]


def test_behavioral_novelty_keeps_labeled_budget_equal_and_reports_descriptor_cost():
    common = dict(
        examples=visible_examples(),
        seed=73,
        population_size=24,
        generations=4,
        elite_count=6,
        max_depth=3,
        viable_mse=1e-12,
    )
    baseline = evolve(strategy="baseline", **common)
    tape = generic_unlabeled_tape(seed=271828, length=16)
    behavioral = evolve(
        strategy="behavioral_novelty",
        novelty_fraction=0.75,
        descriptor_inputs=tape,
        **common,
    )
    assert baseline.evaluations == behavioral.evaluations == 24 * 4
    assert baseline.descriptor_executions == 0
    assert behavioral.descriptor_executions == 24 * 16 * 3
