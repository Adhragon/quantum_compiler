"""
Phase 3 evaluation: route the 5 benchmarks with each trained RL checkpoint
(plus the greedy baseline) and score them with the same metrics as Phase 1.

Pipeline per circuit (mirrors SABRE at optimization_level=3 so the comparison
is routing-vs-routing):
  1. initial layout = SABRE's layout for that circuit (shared with SABRE)
  2. routing        = RL policy (or greedy) inserts SWAPs
  3. post-routing   = Qiskit O3 translation + optimization with layout fixed
                      and routing disabled (identical to what SABRE gets
                      after its own routing)
compile_time_s covers steps 2+3 (SABRE's covers layout + routing + 3).

RUN (from backend/, after training):
    python -m rl.evaluate                         # every rl/checkpoints/*.zip
    python -m rl.evaluate --checkpoints rl/checkpoints/sb3_agent.zip
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from config import (BASELINE_CSV, RL_RESULTS_CSV, SABRE_OPT_LEVEL, SEED, get_backend,
                    load_benchmark_circuits)
from evaluation.metrics import estimated_fidelity, state_fidelity_vs_ideal, total_gate_count
from rl.circuit_utils import circuit_to_task, sabre_layout_and_swaps
from rl.env import RoutingEnv
from rl.train import route


def post_routing_pm(backend):
    """SABRE's O3 pipeline minus layout search and routing: the routed circuit
    already lives on physical qubits, so pin the trivial layout and only check
    connectivity. init is dropped because its ElidePermutations pass would
    delete the inserted SWAPs and break connectivity."""
    pm = generate_preset_pass_manager(
        optimization_level=SABRE_OPT_LEVEL, backend=backend,
        initial_layout=list(range(backend.num_qubits)),
        routing_method="none", seed_transpiler=SEED,
    )
    pm.init = None
    return pm


def score(name, method, qc, compiled, swaps, compile_time, backend, notes=""):
    sf, leak = state_fidelity_vs_ideal(qc, compiled)
    return {
        "circuit": name, "method": method,
        "depth": compiled.depth(), "swaps": swaps,
        "total_gates": total_gate_count(compiled),
        "compile_time_s": compile_time,
        "estimated_fidelity": estimated_fidelity(compiled, backend),
        "state_fidelity": sf, "leakage": leak, "notes": notes,
    }


def evaluate_policy(method, policy, env, bench, pm, backend):
    """policy(env, task, layout) -> swaps; must leave env holding the routed episode."""
    rows = []
    for name, (qc, task, layout) in bench.items():
        t0 = time.perf_counter()
        swaps = policy(env, task, layout)
        compiled = pm.run(env.routed_circuit())
        dt = time.perf_counter() - t0
        notes = "shortest-path fallback used" if env.used_fallback else ""
        rows.append(score(name, method, qc, compiled, swaps, dt, backend, notes))
    return rows


def greedy_policy(env, task, layout):
    env.reset(options={"task": task, "layout": layout})
    while not env.done and env.num_swaps < env.max_swaps:
        env.step(env.greedy_action())
    env.complete_with_shortest_paths()
    return env.num_swaps


def load_model(path):
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    masked = "nomask" not in os.path.basename(path)
    return (MaskablePPO if masked else PPO).load(path, device="cpu"), masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="default: every .zip in rl/checkpoints/")
    args = ap.parse_args()
    paths = args.checkpoints or sorted(glob.glob("rl/checkpoints/*.zip"))

    backend = get_backend()
    circuits = load_benchmark_circuits()
    bench, sabre_swaps = {}, {}
    for name, qc in circuits.items():
        layout, sabre_swaps[name] = sabre_layout_and_swaps(qc, backend, SEED)
        bench[name] = (qc, circuit_to_task(qc, name, backend), layout)

    env = RoutingEnv(backend, tasks=[t for _, t, _ in bench.values()])
    pm = post_routing_pm(backend)

    rows = evaluate_policy("GREEDY", greedy_policy, env, bench, pm, backend)
    for path in paths:
        model, masked = load_model(path)
        method = "RL_" + os.path.basename(path).removesuffix(".zip")
        policy = lambda e, t, l, m=model, k=masked: route(m, e, t, l, k)
        rows += evaluate_policy(method, policy, env, bench, pm, backend)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RL_RESULTS_CSV), exist_ok=True)
    df.to_csv(RL_RESULTS_CSV, index=False)

    # ---- summary: SABRE / QMAP from Phase 1 next to greedy + RL ----
    pd.set_option("display.width", 160)
    cols = ["depth", "swaps", "total_gates", "compile_time_s", "estimated_fidelity", "state_fidelity"]
    full = df
    if os.path.exists(BASELINE_CSV):
        full = pd.concat([pd.read_csv(BASELINE_CSV), df], ignore_index=True)
    print("\n=== per circuit ===")
    print(full[["circuit", "method"] + cols].to_string(index=False))

    print("\n=== totals over 5 circuits (RL = mean +/- std across checkpoints) ===")
    per_method = full.groupby("method", sort=False)[cols].agg(
        {"depth": "sum", "swaps": "sum", "total_gates": "sum", "compile_time_s": "sum",
         "estimated_fidelity": "mean", "state_fidelity": "min"})
    rl = per_method[per_method.index.str.startswith("RL_")]
    print(per_method[~per_method.index.str.startswith("RL_")].to_string())
    if len(rl):
        summary = {c: f"{rl[c].mean():.4g} +/- {rl[c].std(ddof=0):.2g}" for c in cols}
        print(pd.DataFrame([summary], index=[f"RL ({len(rl)} ckpt)"]).to_string())
    print("(estimated_fidelity = mean over circuits, state_fidelity = worst circuit)")
    print(f"\nSABRE routing-only SWAPs (same layouts): {sabre_swaps}")
    print(f"Saved {RL_RESULTS_CSV}")


if __name__ == "__main__":
    main()
