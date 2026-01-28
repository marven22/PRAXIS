from __future__ import annotations

from typing import Any

import numpy as np


def entropy_from_counts(counts: np.ndarray) -> float:
    total = max(1, counts.sum())
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def kl_from_counts(counts_p: np.ndarray, counts_q: np.ndarray, eps: float = 1e-12) -> float:
    p = counts_p.astype(np.float64) + eps
    q = counts_q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
