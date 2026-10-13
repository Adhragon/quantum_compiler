"""
Phase 3 + 4 evaluation: route the 5 benchmarks with every trained agent (plus
the greedy baseline) and score them with the same metrics as Phase 1.

Pipeline per circuit (mirrors SABRE at optimization_level=3 so the comparison
is routing-vs-routing):
  1. initial layout = SABRE's layout, the GNN's best-predicted layout (4B), or
                      the GNN's top-20 layouts, each routed, fewest SWAPs kept
                      (ties: lower depth; time covers every candidate)
  2. routing        = RL policy (or greedy) inserts SWAPs
  3. post-routing   = Qiskit O3 translation + optimization with layout fixed
                      and routing disabled (identical to what SABRE gets
                      after its own routing)
compile_time_s covers steps 2+3, plus GNN layout scoring for GNN-layout rows
(SABRE's covers layout + routing + 3).

Agents evaluated:
  rl/checkpoints/*.zip       Phase 3 agents (observation = routing state)
  gnn_rl/checkpoints/*.zip   Phase 4A agents (observation + GNN embeddings)

Method names: <policy>[+GNN_layout | +GNN_top20_layout], e.g. RL_sb3_agent_seed1+GNN_layout.

RUN (from backend/, after training):
    python -m rl.evaluate                    # all checkpoints, both layouts
    python -m rl.evaluate --no-gnn-layout    # SABRE layout only (no trained GNN needed)
    python -m rl.evaluate --checkpoints rl/checkpoints/sb3_agent.zip
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

import pandas as pd
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from config import (BASELINE_CSV, RESULTS_DIR, RL_RESULTS_CSV, SABRE_OPT_LEVEL, SEED,
                    get_backend, load_benchmark_circuits)
from evaluation.metrics import estimated_fidelity, state_fidelity_vs_ideal, total_gate_count
from rl.circuit_utils import circuit_to_task, sabre_layout_and_swaps
from rl.env import RoutingEnv
from rl.train import route

GNN_RESULTS_CSV = f"{RESULTS_DIR}/gnn_results.csv"
TOP_K = 20  # same shortlist size as gnn.evaluate_shared


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


def evaluate_policy(method, policy, env, bench, pm, backend, layout_key):
    """policy(env, task, layout) -> swaps; must leave env holding the routed episode.
    bench[name] = (qc, task, {layout_key: (candidate layouts, layout_seconds)}).
    Every candidate is routed; the one with the fewest SWAPs (ties: lower
    depth) is reported, and compile time covers all of them."""
    rows = []
    for name, (qc, task, layouts) in bench.items():
        candidates, layout_time = layouts[layout_key]
        t0 = time.perf_counter()
        best = None
        for rank, layout in enumerate(candidates, start=1):
            swaps = policy(env, task, layout)
            compiled = pm.run(env.routed_circuit())
            key = (swaps, compiled.depth())
            if best is None or key < best[0]:
                best = (key, swaps, compiled, layout, rank, env.used_fallback)
        dt = time.perf_counter() - t0 + layout_time
        _, swaps, compiled, layout, rank, fallback = best
        notes = f"layout={layout}" + (f" gnn_rank={rank}" if len(candidates) > 1 else "")
        notes += "; shortest-path fallback used" if fallback else ""
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


def variant(method: str) -> str:
    """Strip the seed so seeds of one configuration are aggregated together."""
    return re.sub(r"_seed\d+", "", method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="default: every .zip in rl/checkpoints/ and gnn_rl/checkpoints/")
    ap.add_argument("--no-gnn-layout", action="store_true",
                    help="skip Phase 4B (routing from the GNN-selected layout)")
    args = ap.parse_args()
    paths = args.checkpoints or sorted(glob.glob("rl/checkpoints/*.zip")) + sorted(glob.glob("gnn_rl/checkpoints/*.zip"))
    gnn_obs_needed = any("gnn_rl" in p.replace("\\", "/") for p in paths)

    backend = get_backend()
    circuits = load_benchmark_circuits()

    selector = None
    if not args.no_gnn_layout or gnn_obs_needed:
        from gnn.gnn_layout import GNNLayoutSelector
        selector = GNNLayoutSelector(backend)

    bench, sabre_swaps = {}, {}
    for name, qc in circuits.items():
        layout, sabre_swaps[name] = sabre_layout_and_swaps(qc, backend, SEED)
        layouts = {"SABRE": ([layout], 0.0)}
        if not args.no_gnn_layout:
            top, _, gnn_time = selector.top_k(qc, TOP_K)
            layouts["GNN"] = (top[:1], gnn_time)
            layouts[f"GNN_top{TOP_K}"] = (top, gnn_time)
        bench[name] = (qc, circuit_to_task(qc, name, backend), layouts)
    layout_keys = list(next(iter(bench.values()))[2])

    tasks = [t for _, t, _ in bench.values()]
    env = RoutingEnv(backend, tasks=tasks)
    gnn_env = None
    if gnn_obs_needed:
        from gnn_rl.features import GNNObservation
        feat = GNNObservation(backend)
        gnn_env = RoutingEnv(backend, tasks=tasks, extra_obs_dim=feat.dim, extra_obs_fn=feat)
    pm = post_routing_pm(backend)

    def suffix(key):
        return "" if key == "SABRE" else f"+{key}_layout"

    rows = []
    for key in layout_keys:
        rows += evaluate_policy("GREEDY" + suffix(key), greedy_policy, env, bench, pm, backend, key)
    for path in paths:
        model, masked = load_model(path)
        is_gnn_obs = "gnn_rl" in path.replace("\\", "/")
        base = ("RL_GNNobs_" if is_gnn_obs else "RL_") + os.path.basename(path).removesuffix(".zip")
        policy = lambda e, t, l, m=model, k=masked: route(m, e, t, l, k)
        for key in layout_keys:
            rows += evaluate_policy(base + suffix(key), policy, gnn_env if is_gnn_obs else env,
                                    bench, pm, backend, key)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RL_RESULTS_CSV), exist_ok=True)
    df.to_csv(RL_RESULTS_CSV, index=False)

    # ---- summary next to SABRE / QMAP (Phase 1) and GNN layout + SABRE (Phase 2) ----
    pd.set_option("display.width", 180)
    full = pd.concat([pd.read_csv(p) for p in (BASELINE_CSV, GNN_RESULTS_CSV) if os.path.exists(p)] + [df],
                     ignore_index=True)
    cols = ["depth", "swaps", "total_gates", "compile_time_s", "estimated_fidelity", "state_fidelity"]
    per_method = full.groupby("method", sort=False).agg(
        depth=("depth", "sum"), swaps=("swaps", "sum"), total_gates=("total_gates", "sum"),
        compile_time_s=("compile_time_s", "sum"), estimated_fidelity=("estimated_fidelity", "mean"),
        state_fidelity=("state_fidelity", "min"))
    per_method["variant"] = [variant(m) for m in per_method.index]
    summary = per_method.groupby("variant", sort=False)[cols].agg(["mean", "std"])
    counts = per_method.groupby("variant", sort=False).size()
    table = pd.DataFrame({
        c: [f"{m:.4g}" + (f" +/- {s:.2g}" if counts[v] > 1 else "")
            for v, (m, s) in zip(summary.index, summary[c].fillna(0).itertuples(index=False))]
        for c in cols}, index=[f"{v} (n={counts[v]})" if counts[v] > 1 else v for v in summary.index])
    print("\n=== totals over 5 circuits (n>1: mean +/- std over seeds) ===")
    print(table.to_string())
    print("(estimated_fidelity = mean over circuits, state_fidelity = worst circuit)")

    swaps = per_method.groupby("variant", sort=False)["swaps"].mean()
    grid = {"SABRE layout": {"SABRE routing": swaps.get("SABRE"), "RL routing": swaps.get("RL_sb3_agent"),
                             "RL + GNN obs (4A)": swaps.get("RL_GNNobs_sb3_agent_gnn")},
            "GNN layout": {"SABRE routing": swaps.get("GNN_layout+SABRE"),
                           "RL routing": swaps.get("RL_sb3_agent+GNN_layout"),
                           "RL + GNN obs (4A)": swaps.get("RL_GNNobs_sb3_agent_gnn+GNN_layout")},
            f"GNN top-{TOP_K} layouts": {
                "SABRE routing": swaps.get(f"GNN_top{TOP_K}+SABRE"),
                "RL routing": swaps.get(f"RL_sb3_agent+GNN_top{TOP_K}_layout"),
                "RL + GNN obs (4A)": swaps.get(f"RL_GNNobs_sb3_agent_gnn+GNN_top{TOP_K}_layout")}}
    print("\n=== total SWAPs: layout x routing (RL = mean over seeds; blank = not run) ===")
    print(pd.DataFrame(grid).T.to_string(na_rep=""))
    print(f"\nSABRE routing-only SWAPs (SABRE layouts): {sabre_swaps}")
    print(f"Saved {RL_RESULTS_CSV}")


if __name__ == "__main__":
    main()
