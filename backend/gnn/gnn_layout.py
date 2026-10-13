"""
Bridge from the Phase 2 GNN project (backend/gnn/src/qubit_allocation) to the
rest of the backend: pick an initial layout for any circuit with the trained
scorer. Used by gnn.evaluate_shared and, in Phase 4, by the RL pipeline.

    from gnn.gnn_layout import GNNLayoutSelector
    selector = GNNLayoutSelector()
    layout, predicted_swaps, seconds = selector.select(qc)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

GNN_ROOT = Path(__file__).resolve().parent
if str(GNN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(GNN_ROOT / "src"))

from qubit_allocation.circuits import prepare_for_routing  # noqa: E402
from qubit_allocation.graph_features import build_circuit_graph, build_hardware_graph  # noqa: E402
from qubit_allocation.layout_generator import all_layouts  # noqa: E402
from qubit_allocation.model import MappingAwareLayoutScorer  # noqa: E402
from qubit_allocation.train import _to_model_batch  # noqa: E402

CHECKPOINT = GNN_ROOT / "models" / "best_gnn_scorer.pt"

# The scorer is small: on a 4-thread laptop CPU, PyTorch's default thread pool
# oversubscribes and scoring all 2,520 layouts runs ~3x slower than on 1 thread.
torch.set_num_threads(1)


class GNNLayoutSelector:
    def __init__(self, backend=None, checkpoint: Path = CHECKPOINT, seed: int = 42):
        if backend is None:
            from config import get_backend
            backend = get_backend()
        self.backend = backend
        self.seed = seed
        self.device = torch.device("cpu")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model = MappingAwareLayoutScorer(**ckpt["model_config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.target_mean = float(ckpt["target_mean"])
        self.target_std = float(ckpt["target_std"])
        self.hw_graph, self.distances = build_hardware_graph(backend.coupling_map)

    def predict(self, qc, layouts: list[list[int]], batch_size: int = 512) -> np.ndarray:
        """Predicted inserted-SWAP count for each candidate layout of qc."""
        graph = build_circuit_graph(prepare_for_routing(qc, self.seed))
        out = []
        with torch.no_grad():
            for i in range(0, len(layouts), batch_size):
                norm = _to_model_batch(self.model, graph, self.hw_graph, self.distances,
                                       layouts[i:i + batch_size], self.device)
                out.append(norm.cpu().numpy() * self.target_std + self.target_mean)
        return np.maximum(np.concatenate(out), 0.0)

    def top_k(self, qc, k: int) -> tuple[list[list[int]], np.ndarray, float]:
        """Score every injective layout; return the k with the lowest predicted
        SWAPs (best first), their predictions, and scoring seconds. Ties keep
        enumeration order, matching qubit_allocation.evaluate."""
        t0 = time.perf_counter()
        layouts = [list(l) for l in all_layouts(qc.num_qubits, self.backend.num_qubits)]
        pred = self.predict(qc, layouts)
        order = np.argsort(pred, kind="stable")[:k]
        return [layouts[i] for i in order], pred[order], time.perf_counter() - t0

    def select(self, qc) -> tuple[list[int], float, float]:
        """Best single layout: (layout, predicted SWAPs, scoring seconds)."""
        layouts, pred, seconds = self.top_k(qc, 1)
        return layouts[0], float(pred[0]), seconds
