from evox.program import Node
from evox.transfer import run_transfer_condition
from evox.worlds import followup_examples


def seed_programs():
    direct = [Node("x0"), Node("add", (Node("x0"), Node("const0")))]
    wrong = [Node("mul", (Node("x1"), Node("x2"))), Node("add", (Node("mul", (Node("x1"), Node("x2"))), Node("const0")))]
    return direct, wrong


def test_transfer_conditions_obey_same_fixed_budget():
    direct, wrong = seed_programs()
    kwargs = dict(examples=followup_examples("direct"), population_size=24, generations=5, elite_count=6, max_depth=3, viable_mse=1e-12)
    a = run_transfer_condition("correct", source_programs=direct, seed=101, **kwargs)
    b = run_transfer_condition("wrong", source_programs=wrong, seed=102, **kwargs)
    c = run_transfer_condition("restart", source_programs=None, seed=103, **kwargs)
    assert a.total_evaluations == b.total_evaluations == c.total_evaluations == 120
    assert all(r.evaluations_to_exact is None or r.evaluations_to_exact <= r.total_evaluations for r in (a, b, c))


def test_transfer_replays_deterministically():
    direct, _ = seed_programs()
    kwargs = dict(condition="correct", source_programs=direct, examples=followup_examples("direct"), seed=55, population_size=24, generations=5, elite_count=6, max_depth=3, viable_mse=1e-12)
    a = run_transfer_condition(**kwargs)
    b = run_transfer_condition(**kwargs)
    assert a == b


def test_structure_preserving_mode_seed_adapts_both_reference_modes():
    direct, wrong = seed_programs()
    relational = wrong
    common = dict(population_size=96, generations=5, elite_count=16, max_depth=3, viable_mse=1e-12)
    a = run_transfer_condition("direct", source_programs=direct, examples=followup_examples("direct"), seed=404, **common)
    b = run_transfer_condition("relational", source_programs=relational, examples=followup_examples("relational"), seed=404, **common)
    assert a.success
    assert b.success
