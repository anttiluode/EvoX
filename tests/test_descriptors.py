import numpy as np

from evox.descriptors import behavior_matrix, generic_unlabeled_tape, in_manifold_unlabeled_tape
from evox.program import Node


def _direct() -> Node:
    return Node("x0")


def _relational() -> Node:
    return Node("mul", (Node("x1"), Node("x2")))


def test_generic_unlabeled_tape_is_deterministic_unique_and_unfiltered():
    first = generic_unlabeled_tape(seed=271828, length=32)
    second = generic_unlabeled_tape(seed=271828, length=32)
    assert first == second
    assert len(first) == 32
    assert len(set(first)) == 32
    assert all(all(-3 <= value <= 3 for value in x) for x in first)
    # The generic tape is not disagreement-filtered: it must contain both
    # reference-agreement and reference-disagreement situations.
    agreements = [x for x in first if x[0] == x[1] * x[2]]
    disagreements = [x for x in first if x[0] != x[1] * x[2]]
    assert agreements
    assert disagreements


def test_in_manifold_tape_contains_no_labels_and_collapses_reference_behavior():
    tape = in_manifold_unlabeled_tape(seed=314159, length=32)
    assert len(tape) == 32
    assert len(set(tape)) == 32
    assert all(x[0] == x[1] * x[2] for x in tape)
    behavior = behavior_matrix([_direct(), _relational()], tape)
    np.testing.assert_array_equal(behavior[0], behavior[1])


def test_generic_tape_exposes_reference_behavioral_difference_without_targets():
    tape = generic_unlabeled_tape(seed=271828, length=32)
    behavior = behavior_matrix([_direct(), _relational()], tape)
    assert behavior.shape == (2, 32)
    assert np.any(behavior[0] != behavior[1])


def test_common_descriptor_coordinate_permutation_preserves_pairwise_distance():
    tape = generic_unlabeled_tape(seed=271828, length=32)
    programs = [_direct(), _relational(), Node("x1")]
    behavior = behavior_matrix(programs, tape)
    perm = np.random.default_rng(9).permutation(behavior.shape[1])
    original = np.linalg.norm(behavior[:, None, :] - behavior[None, :, :], axis=2)
    shuffled = np.linalg.norm(behavior[:, None, perm] - behavior[None, :, perm], axis=2)
    np.testing.assert_allclose(original, shuffled)
