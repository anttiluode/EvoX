import math

import numpy as np
import pytest

from evox.active import (
    CategoricalDistribution,
    GaussianDistribution,
    choose_intervention,
    expected_information_gain,
    posterior_update,
)


def test_categorical_posterior_matches_hand_calculation():
    prior = np.asarray([0.5, 0.3, 0.2])
    predictions = [
        CategoricalDistribution({"pass": 0.9, "fail": 0.1}),
        CategoricalDistribution({"pass": 0.2, "fail": 0.8}),
        CategoricalDistribution({"pass": 0.5, "fail": 0.5}),
    ]
    posterior = posterior_update(prior, predictions, "fail")
    expected = np.asarray([0.05, 0.24, 0.10])
    expected /= expected.sum()
    assert np.allclose(posterior, expected)


def test_gaussian_posterior_matches_direct_likelihood():
    prior = np.asarray([0.4, 0.6])
    predictions = [GaussianDistribution(0.0, 0.5), GaussianDistribution(1.0, 0.5)]
    observation = 0.25
    posterior = posterior_update(prior, predictions, observation)

    likelihood = np.asarray(
        [
            math.exp(-0.5 * ((observation - 0.0) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi)),
            math.exp(-0.5 * ((observation - 1.0) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi)),
        ]
    )
    expected = prior * likelihood
    expected /= expected.sum()
    assert np.allclose(posterior, expected)


def test_distributions_reject_invalid_parameters():
    with pytest.raises(ValueError):
        CategoricalDistribution({"a": 0.2, "b": 0.2})
    with pytest.raises(ValueError):
        CategoricalDistribution({"a": -0.1, "b": 1.1})
    with pytest.raises(ValueError):
        GaussianDistribution(0.0, 0.0)


def test_information_gain_is_zero_when_predictions_are_identical():
    prior = np.asarray([0.2, 0.3, 0.5])
    predictions = [CategoricalDistribution({0: 0.25, 1: 0.75}) for _ in range(3)]
    assert expected_information_gain(prior, predictions) == pytest.approx(0.0, abs=1e-12)


def test_active_chooser_selects_uniquely_discriminating_intervention():
    hypotheses = ("h0", "h1", "h2")
    interventions = ("uninformative", "partial", "separate")
    table = {
        "uninformative": {"h0": "x", "h1": "x", "h2": "x"},
        "partial": {"h0": "x", "h1": "y", "h2": "y"},
        "separate": {"h0": "x", "h1": "y", "h2": "z"},
    }

    def predict(hypothesis, intervention):
        return CategoricalDistribution({table[intervention][hypothesis]: 1.0})

    selected, gain = choose_intervention(
        hypotheses,
        interventions,
        predict,
        np.asarray([1 / 3, 1 / 3, 1 / 3]),
    )
    assert selected == "separate"
    assert gain == pytest.approx(math.log2(3), rel=1e-10)


def test_hypothesis_permutation_does_not_change_selected_intervention():
    interventions = ("a", "b", "c")
    table = {
        "a": {"h0": 0, "h1": 0, "h2": 1},
        "b": {"h0": 0, "h1": 1, "h2": 2},
        "c": {"h0": 1, "h1": 1, "h2": 1},
    }

    def predict(hypothesis, intervention):
        return CategoricalDistribution({table[intervention][hypothesis]: 1.0})

    hypotheses_a = ("h0", "h1", "h2")
    hypotheses_b = ("h2", "h0", "h1")
    selected_a, _ = choose_intervention(
        hypotheses_a, interventions, predict, np.asarray([0.2, 0.5, 0.3])
    )
    selected_b, _ = choose_intervention(
        hypotheses_b, interventions, predict, np.asarray([0.3, 0.2, 0.5])
    )
    assert selected_a == selected_b == "b"


def test_gaussian_quadrature_weights_are_normalized():
    dist = GaussianDistribution(mean=2.0, sigma=0.7, quadrature_points=17)
    points = dist.quadrature()
    assert len(points) == 17
    assert sum(weight for _, weight in points) == pytest.approx(1.0, abs=1e-12)
