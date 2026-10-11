"""
Sanity checks for the Phase 3 routing env. Not a benchmark.

RUN (from backend/):
    python -m rl.smoke_test

Checks:
  1. Gymnasium + SB3 env checkers pass.
  2. Random and greedy policies finish every benchmark; every 2q gate in the
     routed circuit lands on a real coupler.
  3. Routed circuits are correct (state fidelity vs ideal == 1).
  4. A few thousand PPO steps run end-to-end.
Also prints SABRE's true SWAP count (counted after routing, before basis
translation decomposes SWAPs into CXs) as a reference.
"""

from __future__ import annotations

import time

import numpy as np
from config import SEED, get_backend, load_benchmark_circuits
from evaluation.metrics import state_fidelity_vs_ideal
from rl.circuit_utils import circuit_to_task, sabre_layout_and_swaps
from rl.env import RoutingEnv


def run_episode(env, policy, options=None):
    obs, info = env.reset(options=options)
    terminated = truncated = env.done
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(policy(env))
    if truncated:
        env.complete_with_shortest_paths()
    return env.num_swaps


def assert_on_couplers(env, qc):
    edges = set(env.edges)
    for instr in qc.data:
        if len(instr.qubits) == 2:
            pair = tuple(sorted(qc.find_bit(q).index for q in instr.qubits))
            assert pair in edges, f"{instr.operation.name} on non-coupler {pair}"


def main():
    backend = get_backend()
    circuits = load_benchmark_circuits()
    tasks = {name: circuit_to_task(qc, name, backend) for name, qc in circuits.items()}
    env = RoutingEnv(backend, tasks=list(tasks.values()))
    rng = np.random.default_rng(SEED)

    print("== 1. env checkers")
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check
    gym_check(RoutingEnv(backend, tasks=list(tasks.values())), skip_render_check=True)
    sb3_check(RoutingEnv(backend, tasks=list(tasks.values())))
    print(f"   ok  obs_dim={env.observation_space.shape[0]}  actions={env.n_edges}  edges={env.edges}")

    print("\n== 2/3. route benchmarks from SABRE's initial layout")
    print(f"   {'circuit':<11}{'2q':>4}{'SABRE':>7}{'greedy':>8}{'random':>8}{'fidelity':>10}")
    for name, qc in circuits.items():
        layout, sabre_swaps = sabre_layout_and_swaps(qc, backend, SEED)
        opts = {"task": tasks[name], "layout": layout}

        rand_swaps = run_episode(env, lambda e: int(rng.choice(np.flatnonzero(e.action_masks()))), opts)
        assert_on_couplers(env, env.routed_circuit())

        greedy_swaps = run_episode(env, lambda e: e.greedy_action(), opts)
        routed = env.routed_circuit()
        assert_on_couplers(env, routed)
        fid, _ = state_fidelity_vs_ideal(qc, routed)

        print(f"   {name:<11}{tasks[name].num_2q:>4}{sabre_swaps:>7}{greedy_swaps:>8}{rand_swaps:>8}{fid:>10.6f}")
        assert fid > 1 - 1e-9, f"{name}: routed circuit is wrong (fidelity {fid})"

    print("\n== 4. PPO + MaskablePPO end-to-end (2048 steps each)")
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    for algo in (PPO, MaskablePPO):
        t0 = time.perf_counter()
        model = algo("MlpPolicy", RoutingEnv(backend, tasks=list(tasks.values())),
                     n_steps=1024, seed=SEED, verbose=0)
        model.learn(total_timesteps=2048)
        dt = time.perf_counter() - t0
        print(f"   {algo.__name__:<12} ok  {2048 / dt:,.0f} steps/s")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
