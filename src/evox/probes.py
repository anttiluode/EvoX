from __future__ import annotations

import math

import numpy as np

from .modes import ModeModel, mode_predictions


def entropy_bits(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=float)
    q = q[q > 0]
    if len(q) == 0:
        return 0.0
    return float(-np.sum(q * np.log2(q)))


def posterior_update(prior: np.ndarray, predictions: np.ndarray, observation: float, sigma: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    logp = np.log(np.clip(prior, 1e-300, 1.0)) - 0.5 * ((observation - predictions) / sigma) ** 2
    logp -= float(np.max(logp))
    p = np.exp(logp)
    return p / p.sum()


def expected_information_gain(prior: np.ndarray, predictions: np.ndarray, sigma: float) -> float:
    prior = np.asarray(prior, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if abs(float(predictions[0] - predictions[1])) <= 1e-12:
        return 0.0
    lo = float(np.min(predictions) - 6.0 * sigma)
    hi = float(np.max(predictions) + 6.0 * sigma)
    grid = np.linspace(lo, hi, 401)
    norm = sigma * math.sqrt(2.0 * math.pi)
    likelihoods = np.exp(-0.5 * ((grid[:, None] - predictions[None, :]) / sigma) ** 2) / norm
    mixture = likelihoods @ prior
    weighted = likelihoods * prior[None, :]
    denom = np.clip(mixture[:, None], 1e-300, None)
    posteriors = weighted / denom
    ent = np.asarray([entropy_bits(row) for row in posteriors])
    expected_entropy = float(np.trapezoid(mixture * ent, grid))
    gain = entropy_bits(prior) - expected_entropy
    return max(0.0, gain)


def choose_active_probe(
    model: ModeModel,
    candidates: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    prior: np.ndarray,
    *,
    used: set[tuple[int, int, int]],
    sigma: float,
) -> tuple[tuple[int, int, int], float]:
    available = sorted(x for x in candidates if x not in used)
    if not available:
        raise ValueError("no unused probes remain")
    best_x = available[0]
    best_gain = -1.0
    for x in available:
        preds = mode_predictions(model, x)
        gain = expected_information_gain(prior, preds, sigma)
        if gain > best_gain + 1e-12:
            best_x, best_gain = x, gain
    return best_x, float(best_gain)
