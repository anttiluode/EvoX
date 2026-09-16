from evox.worlds import visible_examples, counterfactual_pool, target_value


def test_visible_world_hides_direct_vs_relational_mechanism():
    for x, _ in visible_examples():
        assert target_value("direct", x) == target_value("relational", x)


def test_counterfactual_pool_contains_discriminating_inputs():
    pool = counterfactual_pool()
    assert pool
    assert all(target_value("direct", x) != target_value("relational", x) for x in pool)


def test_followup_keeps_core_mode_but_adds_context_term():
    from evox.worlds import followup_examples
    for mode in ("direct", "relational"):
        for x, y in followup_examples(mode):
            assert y == target_value(mode, x) + x[2]
