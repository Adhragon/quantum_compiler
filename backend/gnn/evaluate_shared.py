"""
Phase 2 evaluation in the shared framework: GNN-selected initial layout +
SABRE routing, scored exactly like the Phase 1 baseline (SABRE at
optimization_level=3, true inserted-SWAP count, evaluation/metrics.py incl.
estimated and state fidelity).

qubit_allocation.evaluate (the GNN project's own evaluator) stays as is: it
measures prediction quality and compares layouts at optimization_level=0.
This script produces the numbers that go next to SABRE / QMAP / RL.

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


def main():
    backend = get_backend()
    selector = GNNLayoutSelector(backend)
    rows = []
    for name, qc in load_benchmark_circuits().items():
        layout, predicted, score_time = selector.select(qc)
        compiled, route_time, swaps = run_sabre(qc, backend, initial_layout=layout)
        sf, leak = state_fidelity_vs_ideal(qc, compiled)
        rows.append({
            "circuit": name, "method": "GNN_layout+SABRE",
            "depth": compiled.depth(), "swaps": swaps,
            "total_gates": total_gate_count(compiled),
            "compile_time_s": score_time + route_time,
            "estimated_fidelity": estimated_fidelity(compiled, backend),
            "state_fidelity": sf, "leakage": leak,
            "notes": f"layout={layout} predicted_swaps={predicted:.2f} scoring_s={score_time:.3f}",
        })
        print(f"{name:<11} layout={layout} predicted={predicted:.2f} actual={swaps}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(GNN_RESULTS_CSV, index=False)

    full = pd.concat([pd.read_csv(BASELINE_CSV), df], ignore_index=True) if os.path.exists(BASELINE_CSV) else df
    full = full[full["method"].isin(["SABRE", "MQT_QMAP", "GNN_layout+SABRE"])]
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
