"""Metric helpers shared by training and evaluation."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(yt) != len(yp) or len(yt) == 0:
        raise ValueError("y_true and y_pred must be non-empty and have equal lengths")
    err = yp - yt
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denominator = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = None if denominator == 0 else float(1.0 - np.sum(err ** 2) / denominator)
    return {"count": int(len(yt)), "mae": mae, "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2}


def percentage_reduction(baseline: float, proposed: float) -> float | None:
    if baseline == 0:
        return None
    return (float(baseline) - float(proposed)) / float(baseline) * 100.0


def ranking_metrics(actual, predicted) -> dict[str, float | int | None]:
    """Report Spearman rank correlation and top-5% overlap on a finite set."""
    from scipy.stats import spearmanr
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(a) != len(p) or len(a) == 0:
        raise ValueError("actual and predicted must be non-empty and equally sized")
    corr = spearmanr(a, p).statistic if len(np.unique(a)) > 1 and len(np.unique(p)) > 1 else None
    k = max(1, int(np.ceil(len(a) * 0.05)))
    actual_top = set(np.argsort(a)[:k].tolist())
    predicted_top = set(np.argsort(p)[:k].tolist())
    overlap = len(actual_top & predicted_top) / len(actual_top)
    return {"spearman": None if corr is None else float(corr), "top_5_percent_overlap": float(overlap), "top_k": k}
