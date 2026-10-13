"""
Headroom check: how much estimated fidelity could a noise-aware layout gain
over SABRE on FakeJakartaV2?

For every circuit, route all 2,520 injective initial layouts with SABRE at the
baseline settings (optimization_level=3, layout fixed) and compute the
estimated fidelity. Compares the best/median layout against default SABRE
(which picks its own layout, then VF2PostLayout may relabel it noise-aware).

RUN (from backend/, ~10-20 min on a laptop CPU):
    python -m evaluation.fidelity_headroom
"""

from __future__ import annotations

import itertools
import os
import time

import pandas as pd
from qiskit import transpile

from baseline.benchmark_sabre_qmap import _sabre_kwargs
from config import RESULTS_DIR, get_backend, load_benchmark_circuits
from evaluation.metrics import estimated_fidelity

OUT_CSV = f"{RESULTS_DIR}/fidelity_headroom.csv"


def main():
    backend = get_backend()
    rows = []
    t0 = time.perf_counter()
    for name, qc in load_benchmark_circuits().items():
        default = transpile(qc, backend=backend, **_sabre_kwargs())
        rows.append({"circuit": name, "layout": "SABRE default",
                     "estimated_fidelity": estimated_fidelity(default, backend),
                     "depth": default.depth()})
        for layout in itertools.permutations(range(backend.num_qubits), qc.num_qubits):
            compiled = transpile(qc, backend=backend, **_sabre_kwargs(list(layout)))
            rows.append({"circuit": name, "layout": str(list(layout)),
                         "estimated_fidelity": estimated_fidelity(compiled, backend),
                         "depth": compiled.depth()})
        print(f"{name}: done ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = []
    for name, g in df.groupby("circuit", sort=False):
        sabre = g.loc[g["layout"] == "SABRE default", "estimated_fidelity"].iloc[0]
        fixed = g[g["layout"] != "SABRE default"]["estimated_fidelity"]
        summary.append({
            "circuit": name, "SABRE": sabre, "best_layout": fixed.max(),
            "median_layout": fixed.median(), "worst_layout": fixed.min(),
            "headroom_abs": fixed.max() - sabre,
            "headroom_rel_%": 100 * (fixed.max() - sabre) / sabre,
            "layouts_beating_SABRE": int((fixed > sabre + 1e-12).sum()),
        })
    s = pd.DataFrame(summary)
    pd.set_option("display.width", 160)
    print(s.round(4).to_string(index=False))
    print(f"\nmean SABRE={s.SABRE.mean():.4f}  mean best={s.best_layout.mean():.4f}  "
          f"mean headroom={s['headroom_rel_%'].mean():.2f}%")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
