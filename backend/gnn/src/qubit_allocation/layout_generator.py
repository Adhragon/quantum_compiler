"""Generate valid injective logical-to-physical qubit mappings."""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator


def number_of_layouts(num_logical: int, num_physical: int) -> int:
    if num_logical < 0 or num_physical < 0 or num_logical > num_physical:
        return 0
    return math.perm(num_physical, num_logical)


def all_layouts(num_logical: int, num_physical: int) -> Iterator[tuple[int, ...]]:
    if num_logical > num_physical:
        raise ValueError("Logical qubits cannot outnumber physical qubits.")
    yield from itertools.permutations(range(num_physical), num_logical)


def sample_layouts(num_logical: int, num_physical: int, count: int, seed: int = 42) -> list[tuple[int, ...]]:
    total = number_of_layouts(num_logical, num_physical)
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= total:
        return list(all_layouts(num_logical, num_physical))
    rng = random.Random(seed)
    # Sampling from exhaustive permutations is robust, deterministic, and small
    # for the project's 2,520-candidate primary search space.
    candidates = list(all_layouts(num_logical, num_physical))
    rng.shuffle(candidates)
    return candidates[:count]


def validate_layout(layout: tuple[int, ...] | list[int], num_logical: int, num_physical: int) -> bool:
    return (
        len(layout) == num_logical
        and all(isinstance(q, int) for q in layout)
        and len(set(layout)) == num_logical
        and all(0 <= q < num_physical for q in layout)
    )
