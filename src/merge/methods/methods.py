"""Five merge methods, all operating on dict-of-numpy state dicts.

linear      : per-key weighted average. The trivial baseline.
slerp       : spherical linear interpolation between two parents.
              For >2 parents we sequentially slerp the running result
              with the next parent.
task_arith  : weighted sum of task vectors (parent - base), added back
              to the base. (Ilharco et al. 2022, "Editing Models with
              Task Arithmetic".)
ties        : magnitude-prune the task vectors to top-K%, then
              resolve sign conflicts via majority vote per-element,
              then average the surviving values per element. (Yadav 2023.)
dare        : drop a random fraction p of the task-vector entries,
              rescale the survivors by 1/(1-p), then average. (Yu 2023.)
model_stock : barycentre of fine-tuned parents projected onto the
              hyperplane through the base; in the open-source variant
              it ends up being a closed-form interpolation with weights
              determined by per-parent cosine to the centroid. (Jang 2024.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import StateDict


def linear(parents: list[StateDict], weights: list[float] | None = None) -> StateDict:
    if not parents:
        raise ValueError("need at least one parent")
    if weights is None:
        weights = [1.0 / len(parents)] * len(parents)
    if len(weights) != len(parents):
        raise ValueError("weights length must equal parents length")
    out: StateDict = {}
    for key in parents[0]:
        stack = np.stack([p[key] for p in parents], axis=0)
        w = np.array(weights, dtype=np.float64)
        out[key] = (stack * w.reshape(-1, *([1] * (stack.ndim - 1)))).sum(axis=0)
    return out


def _slerp_pair(a: NDArray[np.float64], b: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    if na < 1e-9 or nb < 1e-9:
        return (1 - t) * a + t * b
    cos = float(np.dot(a_flat, b_flat) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    omega = float(np.arccos(cos))
    if omega < 1e-6:
        return (1 - t) * a + t * b
    sin_omega = float(np.sin(omega))
    coef_a = np.sin((1 - t) * omega) / sin_omega
    coef_b = np.sin(t * omega) / sin_omega
    result: NDArray[np.float64] = (coef_a * a + coef_b * b).astype(np.float64)
    return result


def slerp(parents: list[StateDict], weights: list[float] | None = None) -> StateDict:
    if len(parents) < 2:
        raise ValueError("slerp needs at least two parents")
    if weights is None:
        weights = [1.0 / len(parents)] * len(parents)
    out: StateDict = {}
    for key in parents[0]:
        # progressive slerp; the effective interpolation factor for parent i
        # is its weight summed up to that point
        acc = parents[0][key].astype(np.float64).copy()
        used = float(weights[0])
        for i in range(1, len(parents)):
            w_i = float(weights[i])
            t = w_i / max(used + w_i, 1e-9)
            acc = _slerp_pair(acc, parents[i][key].astype(np.float64), t)
            used += w_i
        out[key] = acc
    return out


def task_arithmetic(
    base: StateDict, parents: list[StateDict], coefficients: list[float] | None = None
) -> StateDict:
    if not parents:
        raise ValueError("need at least one fine-tuned parent")
    if coefficients is None:
        coefficients = [1.0] * len(parents)
    out: StateDict = {}
    for key in base:
        vec = base[key].astype(np.float64).copy()
        for p, c in zip(parents, coefficients, strict=True):
            vec = vec + c * (p[key] - base[key])
        out[key] = vec
    return out


def ties(
    base: StateDict,
    parents: list[StateDict],
    k_percent: float = 0.2,
    coefficients: list[float] | None = None,
) -> StateDict:
    """TIES-Merging (Yadav 2023): magnitude prune top-k%, then sign election."""
    if not parents:
        raise ValueError("need at least one parent")
    if not (0 < k_percent <= 1):
        raise ValueError("k_percent must be in (0, 1]")
    if coefficients is None:
        coefficients = [1.0] * len(parents)
    out: StateDict = {}
    for key in base:
        tvs: list[NDArray[np.float64]] = []
        for p in parents:
            tv = (p[key] - base[key]).astype(np.float64)
            # magnitude prune: keep top-k% by absolute value
            flat = np.abs(tv).reshape(-1)
            if flat.size == 0:
                tvs.append(tv)
                continue
            threshold = np.quantile(flat, 1.0 - k_percent)
            mask = (np.abs(tv) >= threshold).astype(np.float64)
            tvs.append(tv * mask)
        # sign election: per-element, the majority weighted sign wins
        stack = np.stack(tvs, axis=0)
        sign_sum = np.sign(stack).sum(axis=0)
        elected_sign = np.sign(sign_sum)
        # keep only entries whose sign matches the election, then average
        weighted = np.stack([c * tv for c, tv in zip(coefficients, tvs, strict=True)], axis=0)
        agree = (np.sign(stack) == elected_sign[None, ...]).astype(np.float64)
        kept = weighted * agree
        counts = agree.sum(axis=0)
        merged_tv = np.where(counts > 0, kept.sum(axis=0) / np.maximum(counts, 1), 0.0)
        out[key] = base[key] + merged_tv
    return out


def dare(
    base: StateDict,
    parents: list[StateDict],
    drop_p: float = 0.5,
    coefficients: list[float] | None = None,
    seed: int = 7,
) -> StateDict:
    """DARE: drop p of task-vector entries at random, rescale survivors by 1/(1-p)."""
    if not parents:
        raise ValueError("need at least one parent")
    if not (0 <= drop_p < 1):
        raise ValueError("drop_p must be in [0, 1)")
    if coefficients is None:
        coefficients = [1.0 / len(parents)] * len(parents)
    rng = np.random.default_rng(seed)
    out: StateDict = {}
    for key in base:
        merged = np.zeros_like(base[key], dtype=np.float64)
        for p, c in zip(parents, coefficients, strict=True):
            tv = (p[key] - base[key]).astype(np.float64)
            mask = rng.random(tv.shape) >= drop_p
            scaled = (tv * mask) / max(1 - drop_p, 1e-9)
            merged = merged + c * scaled
        out[key] = base[key] + merged
    return out


def model_stock(parents: list[StateDict]) -> StateDict:
    """Model Stock (Jang 2024): closed-form centroid-based interpolation.

    Reference closed-form: for N parents we project the centroid onto each
    parent's direction from the centroid. For N=2 this collapses to the
    midpoint; for N>=3 it gives a slightly inward shrinkage. We implement
    the simplified equal-weight version.
    """
    if len(parents) < 2:
        raise ValueError("model_stock needs >= 2 parents")
    out: StateDict = {}
    for key in parents[0]:
        stack = np.stack([p[key].astype(np.float64) for p in parents], axis=0)
        centroid = stack.mean(axis=0)
        if len(parents) == 2:
            out[key] = centroid
            continue
        # geometric "ratio of cosines" shrinkage approximation
        # (the paper's full derivation needs the base; for the
        # base-less variant we approximate with cosine-to-centroid)
        flats = stack.reshape(stack.shape[0], -1)
        c_flat = centroid.reshape(-1)
        ratios = []
        for i in range(stack.shape[0]):
            a = float(np.linalg.norm(flats[i] - c_flat))
            b = float(np.linalg.norm(flats[i]))
            ratios.append(a / max(b, 1e-9))
        shrink = 1.0 - float(np.mean(ratios)) / max(np.sqrt(len(parents)), 1.0)
        shrink = max(0.0, min(1.0, shrink))
        out[key] = centroid * shrink
    return out
