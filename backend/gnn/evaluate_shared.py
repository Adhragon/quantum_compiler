"""
Phase 2 evaluation in the shared framework: GNN-selected initial layout +
SABRE routing, scored exactly like the Phase 1 baseline (SABRE at
optimization_level=3, true inserted-SWAP count, evaluation/metrics.py incl.
estimated and state fidelity).

qubit_allocation.evaluate (the GNN project's own evaluator) stays as is: it
measures prediction quality and compares layouts at optimization_level=0.
This script produces the numbers that go next to SABRE / QMAP / RL:
  GNN_layout+SABRE        route the GNN's single best-predicted layout
  GNN_top20+SABRE         route its TOP_K best-predicted layouts and keep the
                          one with the fewest SWAPs (ties: lower depth);
                          compile time includes every candidate routing
  GNN_top20_fid+SABRE     same 20 routed candidates, keep the one with the
                          highest estimated fidelity (noise-aware selection
                          from the backend's calibration data; no retraining)

RUN (from backend/, after the GNN is trained):
    python -m gnn.evaluate_shared
"""

from __future__ import annotations

import os

import pandas as pd

from baseline.benchmark_sabre_qmap import run_sabre
from config import BASELINE_CSV, RESULTS_DIR, get_backend, load_benchmark_circuits
from evaluation.metrics import estimated_fidelity, state_fidelity_vs_ideal, total_gate_count
from gnn.gnn_layout import GNNLayoutSelector

GNN_RESULTS_CSV = f"{RESULTS_DIR}/gnn_results.csv"
TOP_K = 20


def main():
    backend = get_backend()
    selector = GNNLayoutSelector(backend)
    rows = []
    for name, qc in load_benchmark_circuits().items():
        layouts, predicted, score_time = selector.top_k(qc, TOP_K)
        candidates = []
        for layout in layouts:
            compiled, route_time, swaps = run_sabre(qc, backend, initial_layout=layout)
            fid = estimated_fidelity(compiled, backend)
            candidates.append((swaps, compiled.depth(), route_time, layout, compiled, fid))
        selections = (
            ("GNN_layout+SABRE", candidates[:1], lambda c: (c[0], c[1])),
            (f"GNN_top{TOP_K}+SABRE", candidates, lambda c: (c[0], c[1])),
            (f"GNN_top{TOP_K}_fid+SABRE", candidates, lambda c: -c[5]),
        )
        for method, pool, key in selections:
            swaps, depth, _, layout, compiled, _ = min(pool, key=key)
            rank = candidates.index(next(c for c in pool if c[3] is layout)) + 1
            sf, leak = state_fidelity_vs_ideal(qc, compiled)
            rows.append({
                "circuit": name, "method": method,
                "depth": depth, "swaps": swaps,
                "total_gates": total_gate_count(compiled),
                "compile_time_s": score_time + sum(c[2] for c in pool),
                "estimated_fidelity": estimated_fidelity(compiled, backend),
                "state_fidelity": sf, "leakage": leak,
                "notes": f"layout={layout} gnn_rank={rank} predicted_swaps={predicted[rank - 1]:.2f} "
                         f"scoring_s={score_time:.3f}",
            })
        print(f"{name:<11} top1: {rows[-3]['swaps']} swaps | top{TOP_K}: {rows[-2]['swaps']} swaps | "
              f"top{TOP_K} by fidelity: {rows[-1]['swaps']} swaps, est. fidelity {rows[-1]['estimated_fidelity']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(GNN_RESULTS_CSV, index=False)

    full = pd.concat([pd.read_csv(BASELINE_CSV), df], ignore_index=True) if os.path.exists(BASELINE_CSV) else df
    full = full[full["method"].isin(["SABRE", "MQT_QMAP", "GNN_layout+SABRE", f"GNN_top{TOP_K}+SABRE",
                                     f"GNN_top{TOP_K}_fid+SABRE"])]
    pd.set_option("display.width", 160)
    print("\n=== swaps per circuit ===")
    print(full.pivot(index="circuit", columns="method", values="swaps").to_string())
    print("\n=== totals over 5 circuits ===")
    print(full.groupby("method", sort=False).agg(
        depth=("depth", "sum"), swaps=("swaps", "sum"), total_gates=("total_gates", "sum"),
        compile_time_s=("compile_time_s", "sum"), estimated_fidelity=("estimated_fidelity", "mean"),
        state_fidelity=("state_fidelity", "min")).to_string())
    print(f"\nSaved {GNN_RESULTS_CSV}")


if __name__ == "__main__":
    main()
