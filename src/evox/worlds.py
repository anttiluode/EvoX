from __future__ import annotations

from itertools import product


def target_value(mode: str, x: tuple[int, int, int], offset: int | float = 0) -> float:
    if mode == "direct":
        base = x[0]
    elif mode == "relational":
        base = x[1] * x[2]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return float(base + offset)


def visible_examples() -> list[tuple[tuple[int, int, int], float]]:
    examples = []
    for x1, x2 in product((-2, -1, 1, 2), repeat=2):
        x = (x1 * x2, x1, x2)
        examples.append((x, target_value("direct", x)))
    return examples


def counterfactual_pool() -> list[tuple[int, int, int]]:
    vals = (-3, -2, -1, 0, 1, 2, 3)
    return [x for x in product(vals, repeat=3) if x[0] != x[1] * x[2]]


def followup_examples(mode: str) -> list[tuple[tuple[int, int, int], float]]:
    points = [
        (-4, -2, 2),
        (4, 2, 2),
        (2, -2, -1),
        (-2, 1, -2),
        (3, 1, 2),
        (-3, -1, 2),
        (1, 2, 1),
        (-1, 2, -2),
        (0, -2, 2),
        (5, 2, 2),
        (-5, -2, 2),
        (2, 2, -1),
    ]
    return [(x, target_value(mode, x) + x[2]) for x in points]
