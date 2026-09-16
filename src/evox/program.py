from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CLIP = 64.0
TERMINALS = ("x0", "x1", "x2", "const_m2", "const_m1", "const0", "const1", "const2")
UNARY = ("neg",)
BINARY = ("add", "sub", "mul")
CONSTANTS = {
    "const_m2": -2.0,
    "const_m1": -1.0,
    "const0": 0.0,
    "const1": 1.0,
    "const2": 2.0,
}


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple["Node", ...] = ()

    def __post_init__(self) -> None:
        if self.op in TERMINALS and self.children:
            raise ValueError(f"terminal {self.op} cannot have children")
        if self.op in UNARY and len(self.children) != 1:
            raise ValueError(f"unary op {self.op} needs one child")
        if self.op in BINARY and len(self.children) != 2:
            raise ValueError(f"binary op {self.op} needs two children")
        if self.op not in TERMINALS + UNARY + BINARY:
            raise ValueError(f"unknown op {self.op}")


def _clip(value: float) -> float:
    return float(np.clip(value, -CLIP, CLIP))


def evaluate(node: Node, x: tuple[int | float, int | float, int | float]) -> float:
    if node.op == "x0":
        return _clip(float(x[0]))
    if node.op == "x1":
        return _clip(float(x[1]))
    if node.op == "x2":
        return _clip(float(x[2]))
    if node.op in CONSTANTS:
        return CONSTANTS[node.op]
    if node.op == "neg":
        return _clip(-evaluate(node.children[0], x))
    left = evaluate(node.children[0], x)
    right = evaluate(node.children[1], x)
    if node.op == "add":
        return _clip(left + right)
    if node.op == "sub":
        return _clip(left - right)
    if node.op == "mul":
        return _clip(left * right)
    raise AssertionError(node.op)


def random_tree(rng: np.random.Generator, max_depth: int) -> Node:
    if max_depth <= 0 or rng.random() < 0.42:
        return Node(str(rng.choice(TERMINALS)))
    if rng.random() < 0.15:
        return Node("neg", (random_tree(rng, max_depth - 1),))
    op = str(rng.choice(BINARY))
    return Node(op, (random_tree(rng, max_depth - 1), random_tree(rng, max_depth - 1)))


def _paths(node: Node, prefix: tuple[int, ...] = ()) -> Iterable[tuple[int, ...]]:
    yield prefix
    for i, child in enumerate(node.children):
        yield from _paths(child, prefix + (i,))


def _replace(node: Node, path: tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    i = path[0]
    children = list(node.children)
    children[i] = _replace(children[i], path[1:], replacement)
    return Node(node.op, tuple(children))


def mutate_tree(node: Node, rng: np.random.Generator, max_depth: int) -> Node:
    paths = tuple(_paths(node))
    path = paths[int(rng.integers(len(paths)))]
    remaining = max(0, max_depth - len(path))
    return _replace(node, path, random_tree(rng, remaining))


def node_count(node: Node) -> int:
    return 1 + sum(node_count(child) for child in node.children)


def to_source(node: Node) -> str:
    if node.op.startswith("const"):
        return str(int(CONSTANTS[node.op]))
    if node.op in ("x0", "x1", "x2"):
        return node.op
    if node.op == "neg":
        return f"(-{to_source(node.children[0])})"
    symbols = {"add": "+", "sub": "-", "mul": "*"}
    return f"({to_source(node.children[0])}{symbols[node.op]}{to_source(node.children[1])})"
