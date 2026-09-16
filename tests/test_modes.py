import numpy as np

from evox.modes import audit_reference_purity, extract_modes
from evox.program import Node
from evox.worlds import counterfactual_pool


def direct_variants():
    x0 = Node("x0")
    z = Node("const0")
    return [x0, Node("add", (x0, z)), Node("sub", (x0, z)), Node("neg", (Node("neg", (x0,)),))]


def relational_variants():
    rel = Node("mul", (Node("x1"), Node("x2")))
    z = Node("const0")
    return [rel, Node("add", (rel, z)), Node("sub", (rel, z)), Node("neg", (Node("neg", (rel,)),))]


def test_behavior_only_svd_separates_reference_families():
    programs = direct_variants() + relational_variants()
    probes = counterfactual_pool()[:40]
    model = extract_modes(programs, probes)
    audit = audit_reference_purity(model)
    assert model.retained_rank >= 1
    assert len(set(model.labels.tolist())) == 2
    assert min(np.bincount(model.labels)) > 0
    assert audit["purity"] >= 0.9


def test_mode_partition_is_invariant_to_row_permutation():
    programs = direct_variants() + relational_variants()
    probes = counterfactual_pool()[:40]
    base = extract_modes(programs, probes)
    perm = np.array([5, 0, 7, 2, 4, 1, 6, 3])
    shuffled = extract_modes([programs[i] for i in perm], probes)
    unshuffled_labels = np.empty_like(shuffled.labels)
    for shuffled_index, original_index in enumerate(perm):
        unshuffled_labels[original_index] = shuffled.labels[shuffled_index]
    base_same = base.labels[:, None] == base.labels[None, :]
    shuffled_same = unshuffled_labels[:, None] == unshuffled_labels[None, :]
    assert np.array_equal(base_same, shuffled_same)
