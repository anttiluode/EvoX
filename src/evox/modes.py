from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .program import Node, evaluate
from .worlds import target_value


@dataclass(frozen=True)
class ModeModel:
    programs: tuple[Node, ...]
    probes: tuple[tuple[int, int, int], ...]
    behavior: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    singular_values: np.ndarray
    retained_rank: int
    explained_variance: float


def behavior_matrix(programs: list[Node] | tuple[Node, ...], probes: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]) -> np.ndarray:
    return np.asarray([[evaluate(program, x) for x in probes] for program in programs], dtype=float)


def _farthest_pair(points: np.ndarray, behavior: np.ndarray) -> tuple[int, int]:
    best: tuple[float, tuple[float, ...], tuple[float, ...], int, int] | None = None
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sum((points[i] - points[j]) ** 2))
            a = tuple(float(v) for v in behavior[i])
            b = tuple(float(v) for v in behavior[j])
            lo, hi = sorted((a, b))
            candidate = (dist, tuple(-v for v in lo), tuple(-v for v in hi), i, j)
            if best is None or candidate > best:
                best = candidate
    if best is None or best[0] <= 1e-15:
        raise ValueError("behavioral population has no separable variance")
    return best[-2], best[-1]


def _kmeans2(points: np.ndarray, behavior: np.ndarray, max_iter: int = 50) -> np.ndarray:
    i, j = _farthest_pair(points, behavior)
    centers = np.stack([points[i].copy(), points[j].copy()])
    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        d0 = np.sum((points - centers[0]) ** 2, axis=1)
        d1 = np.sum((points - centers[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)
        ties = np.isclose(d0, d1, atol=1e-14, rtol=0.0)
        if np.any(ties):
            # Tie-break by behavior signature, not row position.
            for idx in np.where(ties)[0]:
                new_labels[idx] = int(tuple(behavior[idx]) > tuple(np.median(behavior, axis=0)))
        if len(set(new_labels.tolist())) < 2:
            raise ValueError("deterministic k-means collapsed to one cluster")
        new_centers = np.stack([points[new_labels == k].mean(axis=0) for k in (0, 1)])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            break
        labels = new_labels
        centers = new_centers
    return labels


def _canonicalize(labels: np.ndarray, behavior: np.ndarray) -> np.ndarray:
    medians = np.stack([np.median(behavior[labels == k], axis=0) for k in (0, 1)])
    discriminating = np.where(np.abs(medians[0] - medians[1]) > 1e-12)[0]
    if len(discriminating) == 0:
        raise ValueError("clusters do not differ in median behavior")
    j = int(discriminating[0])
    if medians[0, j] <= medians[1, j]:
        return labels
    return 1 - labels


def extract_modes(
    programs: list[Node] | tuple[Node, ...],
    probes: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    variance_threshold: float = 0.95,
) -> ModeModel:
    if len(programs) < 2:
        raise ValueError("need at least two programs")
    if not probes:
        raise ValueError("need at least one probe")
    behavior = behavior_matrix(programs, probes)
    centered = behavior - behavior.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    energy = s**2
    total = float(energy.sum())
    if total <= 1e-15:
        raise ValueError("behavioral population has zero variance")
    cumulative = np.cumsum(energy) / total
    rank = int(np.searchsorted(cumulative, variance_threshold) + 1)
    rank = max(1, min(rank, len(s)))
    embedding = u[:, :rank] * s[:rank]
    labels = _canonicalize(_kmeans2(embedding, behavior), behavior)
    return ModeModel(
        programs=tuple(programs),
        probes=tuple(probes),
        behavior=behavior,
        embedding=embedding,
        labels=labels,
        singular_values=s,
        retained_rank=rank,
        explained_variance=float(cumulative[rank - 1]),
    )


def mode_predictions(model: ModeModel, x: tuple[int, int, int]) -> np.ndarray:
    preds = []
    for k in (0, 1):
        values = [evaluate(program, x) for program, label in zip(model.programs, model.labels) if int(label) == k]
        if not values:
            raise ValueError(f"mode {k} is empty")
        preds.append(float(np.median(values)))
    return np.asarray(preds, dtype=float)


def audit_reference_purity(model: ModeModel) -> dict[str, object]:
    true_family = []
    for row in model.behavior:
        direct = np.asarray([target_value("direct", x) for x in model.probes], dtype=float)
        relational = np.asarray([target_value("relational", x) for x in model.probes], dtype=float)
        direct_mse = float(np.mean((row - direct) ** 2))
        relational_mse = float(np.mean((row - relational) ** 2))
        true_family.append(0 if direct_mse <= relational_mse else 1)
    true = np.asarray(true_family, dtype=int)
    same = float(np.mean(model.labels == true))
    flipped = float(np.mean((1 - model.labels) == true))
    if same >= flipped:
        mapping = {0: "direct", 1: "relational"}
        purity = same
    else:
        mapping = {0: "relational", 1: "direct"}
        purity = flipped
    return {"purity": purity, "cluster_to_reference": mapping}
