"""Deterministic circuit-stratified splits and canonical layout identifiers."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def parse_layout(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Layout must be a JSON array, got {value!r}")
    return [int(q) for q in parsed]


def layout_id(circuit: str, layout_string: str) -> str:
    return f"{circuit}::{json.dumps(parse_layout(layout_string), separators=(',', ':'))}"


def make_split(df: pd.DataFrame, seed: int, train_ratio: float, validation_ratio: float) -> dict[str, list[str]]:
    """Split each circuit's distinct layouts reproducibly into train/validation/test."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Invalid split proportions.")
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "test": []}
    for circuit, group in sorted(df.groupby("circuit", sort=True)):
        row_ids = [layout_id(str(row.circuit), str(row.layout)) for row in group.itertuples(index=False)]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Duplicate circuit-layout identities in {circuit}.")
        order = rng.permutation(len(row_ids))
        n = len(row_ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * validation_ratio))
        n_train = min(n_train, max(1, n - 2))
        n_val = min(n_val, max(1, n - n_train - 1))
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        if not len(train_idx) or not len(val_idx) or not len(test_idx):
            raise ValueError(f"Not enough samples for circuit-stratified train/validation/test split: {circuit} n={n}.")
        result["train"].extend(row_ids[i] for i in train_idx)
        result["validation"].extend(row_ids[i] for i in val_idx)
        result["test"].extend(row_ids[i] for i in test_idx)
    sets = [set(result[k]) for k in ("train", "validation", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Data split leakage detected.")
    if len(set.union(*sets)) != sum(len(s) for s in sets):
        raise AssertionError("Split identity collision detected.")
    return result
