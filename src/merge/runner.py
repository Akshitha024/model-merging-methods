"""Run a sweep of merge methods on synthetic parents and write per-method
JSON + a per-layer drift CSV.

For the in-CI version we generate parents synthetically: each parent is the
base plus a small random task vector. That keeps the suite GPU-free and the
metric ranges meaningful (cosine to base ~ 0.99 for small TVs).

To run on real HF checkpoints, swap _make_parents for a state-dict loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from .methods.methods import dare, linear, model_stock, slerp, ties
from .metrics.compare import cosine, l2_per_layer, mean_abs_drift, top_k_changes
from .types import StateDict


def _make_synthetic(d: int, layers: int = 6, seed: int = 7) -> StateDict:
    rng = np.random.default_rng(seed)
    return {
        f"layer_{i}.weight": rng.standard_normal((d, d)).astype(np.float64) for i in range(layers)
    }


def _add_task_vector(base: StateDict, scale: float, seed: int) -> StateDict:
    rng = np.random.default_rng(seed)
    out: StateDict = {}
    for k, v in base.items():
        out[k] = v + scale * rng.standard_normal(v.shape).astype(np.float64)
    return out


def make_synthetic_problem(
    d: int = 32, layers: int = 6, n_parents: int = 3, scale: float = 0.05
) -> tuple[StateDict, list[StateDict]]:
    base = _make_synthetic(d, layers, seed=0)
    parents = [_add_task_vector(base, scale, seed=i + 1) for i in range(n_parents)]
    return base, parents


def sweep(methods: list[str], out_dir: Path) -> dict[str, dict[str, float]]:
    base, parents = make_synthetic_problem()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    layer_drifts: dict[str, dict[str, float]] = {}

    for m in methods:
        if m == "linear":
            merged = linear(parents)
        elif m == "slerp":
            merged = slerp(parents)
        elif m == "task_arith":
            merged = _alias_task_arith(base, parents)
        elif m == "ties":
            merged = ties(base, parents, k_percent=0.2)
        elif m == "dare":
            merged = dare(base, parents, drop_p=0.5)
        elif m == "model_stock":
            merged = model_stock(parents)
        else:
            raise ValueError(f"unknown method: {m}")

        cos_to_base = cosine(merged, base)
        cos_to_parent_avg = float(np.mean([cosine(merged, p) for p in parents]))
        drift = mean_abs_drift(merged, base)
        layer_d = l2_per_layer(merged, base)
        top = top_k_changes(merged, base, k=5)
        results[m] = {
            "cosine_to_base": cos_to_base,
            "cosine_to_parent_avg": cos_to_parent_avg,
            "mean_abs_drift": drift,
            "n_layers": float(len(layer_d)),
            "top_layer_l2": top[0][1] if top else 0.0,
        }
        layer_drifts[m] = layer_d
        (out_dir / f"{m}__metrics.json").write_text(json.dumps(results[m], indent=2))

    (out_dir / "layer_drifts.json").write_text(json.dumps(layer_drifts, indent=2))
    logger.info("wrote sweep results to {}", out_dir)
    return results


def _alias_task_arith(base: StateDict, parents: list[StateDict]) -> StateDict:
    from .methods.methods import task_arithmetic

    return task_arithmetic(base, parents)
