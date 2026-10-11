"""
Phase 3: train the RL router (MaskablePPO by default) on FakeJakartaV2.

Training mixes random 5-qubit CX circuits with the 5 benchmarks, each with a
random initial layout. Every --eval-freq steps the policy routes the 5
benchmarks from SABRE's initial layout; the checkpoint with the fewest total
SWAPs is saved to config.RL_CHECKPOINT.

RUN (from backend/):
    python -m rl.train                      # 200k steps, ~10-15 min on CPU
    python -m rl.train --timesteps 20000    # quick check
    python -m rl.train --no-mask            # plain PPO, for comparison

Learning curves (optional, separate terminal, from backend/):
    tensorboard --logdir rl/logs
"""

from __future__ import annotations

import argparse
import os
import time

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import RL_CHECKPOINT, SEED, get_backend, load_benchmark_circuits
from rl.circuit_utils import circuit_to_task, sabre_layout_and_swaps
from rl.env import RoutingEnv

LOG_DIR = "rl/logs"


def route(model, env: RoutingEnv, task, layout, masked: bool) -> int:
    """Route one circuit deterministically; returns SWAPs inserted
    (including any shortest-path fallback after truncation)."""
    obs, _ = env.reset(options={"task": task, "layout": layout})
    terminated = truncated = env.done
    while not (terminated or truncated):
        kwargs = {"action_masks": env.action_masks()} if masked else {}
        action, _ = model.predict(obs, deterministic=True, **kwargs)
        obs, _, terminated, truncated, _ = env.step(action)
    if truncated:
        env.complete_with_shortest_paths()
    return env.num_swaps


class BenchmarkEvalCallback(BaseCallback):
    def __init__(self, env, bench, eval_freq, save_path, masked):
        super().__init__()
        self.env, self.bench = env, bench
        self.eval_freq, self.save_path, self.masked = eval_freq, save_path, masked
        self.best = float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        swaps = {name: route(self.model, self.env, task, layout, self.masked)
                 for name, (task, layout, _) in self.bench.items()}
        total = sum(swaps.values())
        for name, s in swaps.items():
            self.logger.record(f"eval/swaps_{name}", s)
        self.logger.record("eval/total_swaps", total)

        mark = ""
        if total < self.best:
            self.best = total
            self.model.save(self.save_path)
            mark = "  <- saved"
        per = "  ".join(f"{n}={s}" for n, s in swaps.items())
        print(f"[{self.num_timesteps:>8}] total={total:<3} {per}{mark}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-mask", action="store_true", help="plain PPO without action masking")
    args = ap.parse_args()
    masked = not args.no_mask

    backend = get_backend()
    circuits = load_benchmark_circuits()
    tasks = {name: circuit_to_task(qc, name, backend) for name, qc in circuits.items()}

    bench = {}
    for name, qc in circuits.items():
        layout, sabre_swaps = sabre_layout_and_swaps(qc, backend, SEED)
        bench[name] = (tasks[name], layout, sabre_swaps)
    sabre_total = sum(s for *_, s in bench.values())
    print("SABRE reference: total=%d  %s" % (
        sabre_total, "  ".join(f"{n}={s}" for n, (*_, s) in bench.items())))

    train_env = Monitor(RoutingEnv(backend, tasks=list(tasks.values())))
    train_env.reset(seed=args.seed)
    eval_env = RoutingEnv(backend, tasks=list(tasks.values()))

    algo = MaskablePPO if masked else PPO
    run_name = f"{algo.__name__}_seed{args.seed}"
    suffix = ("" if masked else "_nomask") + ("" if args.seed == SEED else f"_seed{args.seed}")
    save_path = RL_CHECKPOINT.replace(".zip", f"{suffix}.zip")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = algo(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
        ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=LOG_DIR, seed=args.seed, verbose=0,
    )
    cb = BenchmarkEvalCallback(eval_env, bench, args.eval_freq, save_path, masked)

    print(f"training {run_name} for {args.timesteps:,} steps -> {save_path}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name=run_name)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min  "
          f"best total SWAPs={cb.best} (SABRE={sabre_total})")


if __name__ == "__main__":
    main()
