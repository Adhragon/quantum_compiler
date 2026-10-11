import json
import pandas as pd
from qubit_allocation.split_utils import make_split


def test_train_validation_test_splits_are_disjoint_and_complete():
    rows = []
    import itertools
    all_five_qubit_layouts = list(itertools.permutations(range(5)))
    for circuit in ("dj_5q", "ghz_5q", "graphstate_5q"):
        for i in range(40):
            layout = list(all_five_qubit_layouts[i])
            rows.append({"circuit": circuit, "layout": json.dumps(layout, separators=(",", ":"))})
    df = pd.DataFrame(rows)
    splits = make_split(df, seed=42, train_ratio=0.70, validation_ratio=0.15)
    sets = [set(splits["train"]), set(splits["validation"]), set(splits["test"])]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert len(set.union(*sets)) == len(df)
    assert len(splits["train"]) == 84
    assert len(splits["validation"]) == 18
    assert len(splits["test"]) == 18
