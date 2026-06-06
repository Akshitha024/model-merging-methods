"""Compare merged state dicts to parents and base.

Distinct metrics from prior projects (not RAG-style nDCG/F1):
  - param-wise cosine similarity (flattened)
  - L2 distance per layer
  - mean absolute drift (per parameter)
  - top-K largest parameter changes
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import StateDict


def flat(state: StateDict) -> NDArray[np.float64]:
    parts = [v.reshape(-1).astype(np.float64) for v in state.values()]
    return np.concatenate(parts) if parts else np.zeros(0)


def cosine(a: StateDict, b: StateDict) -> float:
    fa, fb = flat(a), flat(b)
    if fa.size == 0 or fb.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(fa)), float(np.linalg.norm(fb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def l2_per_layer(a: StateDict, b: StateDict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in a:
        if k not in b:
            continue
        diff = a[k].astype(np.float64) - b[k].astype(np.float64)
        out[k] = float(np.linalg.norm(diff))
    return out


def mean_abs_drift(a: StateDict, b: StateDict) -> float:
    diffs = []
    for k in a:
        if k not in b:
            continue
        diffs.append(np.abs(a[k] - b[k]).mean())
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def top_k_changes(a: StateDict, b: StateDict, k: int = 10) -> list[tuple[str, float]]:
    rows = l2_per_layer(a, b)
    return sorted(rows.items(), key=lambda x: x[1], reverse=True)[:k]
