"""
Phase 3: Gymnasium environment for RL-based qubit routing on FakeJakartaV2.

Formulation (Pozzi et al. 2022 style, simplified):
  * Initial layout is fixed per episode (random during training, SABRE's
    layout at evaluation), so the agent only decides routing.
  * Action  = index into coupling_edges(backend): insert a SWAP on that coupler.
  * After every SWAP all gates whose dependencies are met and whose qubits
    are now adjacent are executed automatically.
  * Episode ends when every gate has executed (terminated) or after
    max_swaps SWAPs (truncated; finish with complete_with_shortest_paths()).

Observation (flat float32 vector, fixed length so Phase 4 can append GNN
embeddings via extra_obs_fn):
  [ layout (n_phys) | lookahead excess distances (lookahead) |
    per-edge front-layer distance delta (n_edges) |
    per-edge lookahead distance delta (n_edges) |
    last action one-hot (n_edges) | progress (1) | extra_obs (extra_obs_dim) ]
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate

from config import CIRCUIT_SIZE, get_backend, load_benchmark_circuits
from rl.circuit_utils import (
    RoutingTask,
    circuit_to_task,
    coupling_edges,
    distance_matrix,
    random_task,
    shortest_path,
)


class RoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        backend=None,
        tasks: list[RoutingTask] | None = None,
        num_logical: int = CIRCUIT_SIZE,
        random_circuit_prob: float = 0.5,
        random_num_2q: tuple[int, int] = (5, 30),
        lookahead: int = 8,
        lookahead_decay: float = 0.7,
        max_swaps: int = 50,
        swap_penalty: float = 1.0,
        gate_reward: float = 1.0,
        distance_coef: float = 0.5,
        done_bonus: float = 5.0,
        extra_obs_dim: int = 0,
        extra_obs_fn: Callable[[RoutingTask], np.ndarray] | None = None,
    ):
        super().__init__()
        backend = backend or get_backend()
        self.n_phys = backend.num_qubits
        self.edges = coupling_edges(backend)
        self.n_edges = len(self.edges)
        self.dist = distance_matrix(self.n_phys, self.edges)
        self.dmax = int(self.dist.max())

        if tasks is None:
            circuits = load_benchmark_circuits(size=num_logical)
            tasks = [circuit_to_task(qc, name, backend) for name, qc in circuits.items()]
        self.tasks = tasks
        self.num_logical = num_logical
        self.random_circuit_prob = random_circuit_prob
        self.random_num_2q = random_num_2q

        self.lookahead = lookahead
        self.look_w = lookahead_decay ** np.arange(lookahead)
        self.max_swaps = max_swaps
        self.swap_penalty = swap_penalty
        self.gate_reward = gate_reward
        self.distance_coef = distance_coef
        self.done_bonus = done_bonus

        self.extra_obs_dim = extra_obs_dim
        self.extra_obs_fn = extra_obs_fn

        obs_dim = self.n_phys + lookahead + 3 * self.n_edges + 1 + extra_obs_dim
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_edges)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """options (all optional, used at evaluation time):
             task:   RoutingTask to route instead of sampling one
             layout: list[int], physical qubit for each logical qubit
        """
        super().reset(seed=seed)
        options = options or {}
        fixed_task = options.get("task")

        for _ in range(100):
            task = fixed_task or self._sample_task()
            self._load(task, options.get("layout"))
            # A sampled circuit that is already fully routed teaches nothing.
            if fixed_task is not None or not self.done:
                break

        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        reward = -self.swap_penalty
        delta_front, _ = self._edge_deltas()
        reward -= self.distance_coef * delta_front[action]

        self._apply_swap(action)
        n_exec = self._execute_ready()
        reward += self.gate_reward * n_exec

        terminated = self.done
        if terminated:
            reward += self.done_bonus
        truncated = (not terminated) and self.num_swaps >= self.max_swaps
        return self._obs(), float(reward), terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """For sb3-contrib MaskablePPO: only SWAPs touching a qubit of a
        front-layer gate, and never immediately undoing the previous SWAP."""
        front_phys = {int(self.l2p[q]) for g in self._front() for q in self.task.ops[g][1]}
        mask = np.array([a in front_phys or b in front_phys for a, b in self.edges])
        if self.last_action is not None and mask.sum() > 1:
            mask[self.last_action] = False
        if not mask.any():
            mask[:] = True
        return mask

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    def _sample_task(self) -> RoutingTask:
        if not self.tasks or self.np_random.random() < self.random_circuit_prob:
            lo, hi = self.random_num_2q
            n2q = int(self.np_random.integers(lo, hi + 1))
            return random_task(self.num_logical, n2q, self.np_random)
        return self.tasks[int(self.np_random.integers(len(self.tasks)))]

    def _load(self, task: RoutingTask, layout: list[int] | None):
        self.task = task
        if layout is None:
            # Random placement of all n_phys virtual qubits (logical + ancilla).
            self.l2p = self.np_random.permutation(self.n_phys)
        else:
            layout = [int(p) for p in layout]
            free = [p for p in range(self.n_phys) if p not in layout]
            self.l2p = np.array(layout + free)
        self.p2l = np.empty(self.n_phys, dtype=np.int64)
        self.p2l[self.l2p] = np.arange(self.n_phys)
        self.initial_layout = [int(p) for p in self.l2p[: task.num_logical]]

        # Per-logical-qubit queue of op indices; head[q] points at the next one.
        self.queues = [[] for _ in range(self.n_phys)]
        for i, (_, qs) in enumerate(task.ops):
            for q in qs:
                self.queues[q].append(i)
        self.head = [0] * self.n_phys
        self.executed = np.zeros(len(task.ops), dtype=bool)
        self.twoq_idx = [i for i, (_, qs) in enumerate(task.ops) if len(qs) == 2]
        self.n_2q_done = 0

        self.out: list[tuple] = []  # (operation, physical qubits)
        self.num_swaps = 0
        self.last_action = None
        self.used_fallback = False
        self._execute_ready()

    @property
    def done(self) -> bool:
        return bool(self.executed.all())

    def _is_head(self, i: int) -> bool:
        return all(
            self.head[q] < len(self.queues[q]) and self.queues[q][self.head[q]] == i
            for q in self.task.ops[i][1]
        )

    def _execute_ready(self) -> int:
        """Execute every dependency-free gate that is executable under the
        current layout. Returns the number of 2q gates executed."""
        n_2q = 0
        progress = True
        while progress:
            progress = False
            for q in range(self.n_phys):
                if self.head[q] >= len(self.queues[q]):
                    continue
                i = self.queues[q][self.head[q]]
                if not self._is_head(i):
                    continue
                op, qs = self.task.ops[i]
                phys = tuple(int(self.l2p[x]) for x in qs)
                if len(qs) == 2 and self.dist[phys[0], phys[1]] != 1:
                    continue
                self.out.append((op, phys))
                self.executed[i] = True
                for x in qs:
                    self.head[x] += 1
                if len(qs) == 2:
                    n_2q += 1
                progress = True
        self.n_2q_done += n_2q
        return n_2q

    def _apply_swap(self, action: int):
        a, b = self.edges[action]
        la, lb = self.p2l[a], self.p2l[b]
        self.p2l[a], self.p2l[b] = lb, la
        self.l2p[la], self.l2p[lb] = b, a
        self.out.append((SwapGate(), (a, b)))
        self.num_swaps += 1
        self.last_action = action

    def _front(self) -> list[int]:
        """Unexecuted 2q gates whose dependencies are all satisfied."""
        front = set()
        for q in range(self.n_phys):
            if self.head[q] < len(self.queues[q]):
                i = self.queues[q][self.head[q]]
                if len(self.task.ops[i][1]) == 2 and self._is_head(i):
                    front.add(i)
        return sorted(front)

    def _lookahead_gates(self) -> list[int]:
        out = []
        for i in self.twoq_idx:
            if not self.executed[i]:
                out.append(i)
                if len(out) == self.lookahead:
                    break
        return out

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _excess(self, l2p: np.ndarray, gates: list[int]) -> np.ndarray:
        """Hops beyond adjacency for each gate (0 == executable)."""
        if not gates:
            return np.zeros(0)
        pairs = np.array([self.task.ops[i][1] for i in gates])
        return self.dist[l2p[pairs[:, 0]], l2p[pairs[:, 1]]] - 1

    def _edge_deltas(self) -> tuple[np.ndarray, np.ndarray]:
        """Change in (front-layer, decayed lookahead) excess distance if each
        coupler were swapped. Negative == the swap helps."""
        front, look = self._front(), self._lookahead_gates()
        w = self.look_w[: len(look)]
        f0 = self._excess(self.l2p, front).sum()
        l0 = (self._excess(self.l2p, look) * w).sum()
        d_front = np.zeros(self.n_edges)
        d_look = np.zeros(self.n_edges)
        for e, (a, b) in enumerate(self.edges):
            l2p = self.l2p.copy()
            la, lb = self.p2l[a], self.p2l[b]
            l2p[la], l2p[lb] = b, a
            d_front[e] = self._excess(l2p, front).sum() - f0
            d_look[e] = (self._excess(l2p, look) * w).sum() - l0
        if len(w):
            d_look /= w.sum()
        return d_front, d_look

    def _obs(self) -> np.ndarray:
        layout = self.l2p / (self.n_phys - 1)

        look = np.full(self.lookahead, -1.0)  # -1 == no gate in this slot
        exc = self._excess(self.l2p, self._lookahead_gates())
        look[: len(exc)] = exc / max(self.dmax - 1, 1)

        d_front, d_look = self._edge_deltas()
        d_front = np.clip(d_front / 2.0, -1, 1)

        last = np.zeros(self.n_edges)
        if self.last_action is not None:
            last[self.last_action] = 1.0

        progress = [self.n_2q_done / max(len(self.twoq_idx), 1)]

        parts = [layout, look, d_front, d_look, last, progress]
        if self.extra_obs_dim:
            extra = np.asarray(self.extra_obs_fn(self.task), dtype=np.float64)
            parts.append(np.clip(extra.reshape(self.extra_obs_dim), -1, 1))
        return np.concatenate(parts).astype(np.float32)

    def _info(self) -> dict:
        return {"num_swaps": self.num_swaps, "task": self.task.name, "done": self.done}

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def greedy_action(self) -> int:
        """Baseline policy: best front-layer improvement, ties broken by
        lookahead improvement, restricted to the action mask."""
        d_front, d_look = self._edge_deltas()
        score = d_front + 0.1 * d_look
        score[~self.action_masks()] = np.inf
        return int(np.argmin(score))

    def complete_with_shortest_paths(self):
        """Fallback after truncation: walk the first front gate's qubits
        together along a shortest path until the circuit is finished."""
        self.used_fallback = self.used_fallback or not self.done
        while not self.done:
            a, b = self.task.ops[self._front()[0]][1]
            path = shortest_path(self.edges, int(self.l2p[a]), int(self.l2p[b]))
            self._apply_swap(self.edges.index(tuple(sorted(path[:2]))))
            self._execute_ready()

    def routed_circuit(self) -> QuantumCircuit:
        """Physical circuit (n_phys qubits) with the inserted SWAPs, still in
        the pre-routing gate set; basis-translate before scoring."""
        qc = QuantumCircuit(self.n_phys, name=f"{self.task.name}_rl")
        for op, phys in self.out:
            qc.append(op, list(phys))
        return qc

    @property
    def final_layout(self) -> list[int]:
        return [int(p) for p in self.l2p[: self.task.num_logical]]
