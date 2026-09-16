import numpy as np
from evox.program import Node, evaluate, mutate_tree


def test_expression_tree_evaluates_without_eval_and_clips():
    tree = Node("mul", (Node("x1"), Node("x2")))
    assert evaluate(tree, (0, 3, 4)) == 12.0
    huge = Node("mul", (Node("const2"), Node("mul", (Node("const2"), Node("const2")))))
    assert abs(evaluate(huge, (0, 0, 0))) <= 64.0


def test_mutation_returns_executable_tree():
    rng = np.random.default_rng(7)
    tree = Node("x0")
    child = mutate_tree(tree, rng, max_depth=3)
    value = evaluate(child, (2, 3, 4))
    assert np.isfinite(value)
