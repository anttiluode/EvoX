import numpy as np

from evox.modes import extract_modes, mode_predictions
from evox.probes import choose_active_probe, expected_information_gain, posterior_update
from evox.program import Node


def model():
    direct = [Node("x0"), Node("add", (Node("x0"), Node("const0")))]
    relational = [Node("mul", (Node("x1"), Node("x2"))), Node("add", (Node("mul", (Node("x1"), Node("x2"))), Node("const0")))]
    probes = [(0, 1, 2), (2, 1, 2), (3, 1, 2)]
    return extract_modes(direct + relational, probes)


def test_equal_predictions_have_zero_information_gain():
    prior = np.array([0.5, 0.5])
    assert expected_information_gain(prior, np.array([2.0, 2.0]), sigma=0.25) == 0.0


def test_active_probe_chooses_maximally_separating_point():
    m = model()
    candidates = [(2, 1, 2), (0, 1, 2), (3, 1, 2)]
    chosen, gain = choose_active_probe(m, candidates, np.array([0.5, 0.5]), used=set(), sigma=0.25)
    diffs = {x: abs(np.diff(mode_predictions(m, x))[0]) for x in candidates}
    assert diffs[chosen] == max(diffs.values())
    assert gain > 0


def test_informative_observation_moves_posterior_toward_matching_mode():
    m = model()
    x = (0, 1, 2)
    predictions = mode_predictions(m, x)
    correct = int(np.argmax(predictions))
    posterior = posterior_update(np.array([0.5, 0.5]), predictions, predictions[correct], sigma=0.25)
    assert posterior[correct] > 0.99
