"""Five distinct charts for the merge sweep.

Different from prior projects (no nDCG, ROC, RAG charts):
  - method comparison bar chart on cosine-to-base
  - per-layer L2 drift heatmap (methods x layers)
  - per-method scatter of cosine-to-base vs cosine-to-parent-avg
  - top-5 most-changed layers grouped bar
  - parameter magnitude histogram for a chosen method vs base
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _read(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    data: dict[str, Any] = json.loads(p.read_text())
    return data


# 1. Method comparison bar: cosine_to_base + cosine_to_parent_avg
def plot_method_comparison(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for f in sorted(results_dir.glob("*__metrics.json")):
        m = f.stem.removesuffix("__metrics")
        metrics[m] = _read(f)
    if not metrics:
        out.write_bytes(b"")
        return out
    methods = list(metrics)
    cb = [metrics[m].get("cosine_to_base", 0) for m in methods]
    cp = [metrics[m].get("cosine_to_parent_avg", 0) for m in methods]
    x = np.arange(len(methods))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(methods) + 2), 4.5))
    ax.bar(x - width / 2, cb, width, label="cos(merged, base)", color="#1f77b4")
    ax.bar(x + width / 2, cp, width, label="cos(merged, parent avg)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.95, 1.0)
    ax.set_ylabel("cosine similarity")
    ax.set_title("Per-method cosine to base and to parent average")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 2. Per-layer L2 drift heatmap
def plot_layer_drift_heatmap(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    layer_drifts = _read(results_dir / "layer_drifts.json")
    if not layer_drifts:
        out.write_bytes(b"")
        return out
    methods = sorted(layer_drifts.keys())
    layers = sorted({k for m in methods for k in layer_drifts[m]})
    mat = np.zeros((len(methods), len(layers)))
    for i, m in enumerate(methods):
        for j, layer in enumerate(layers):
            mat[i, j] = float(layer_drifts[m].get(layer, 0))
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(layers) + 2), max(3, 0.4 * len(methods) + 2)))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(len(layers)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if mat[i, j] < mat.max() * 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="L2 drift from base")
    ax.set_title("Per-layer L2 drift by method")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 3. Cosine-to-base vs cosine-to-parent scatter
def plot_cosine_scatter(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for f in sorted(results_dir.glob("*__metrics.json")):
        m = f.stem.removesuffix("__metrics")
        metrics[m] = _read(f)
    if not metrics:
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m, mtx in metrics.items():
        x = mtx.get("cosine_to_base", 0)
        y = mtx.get("cosine_to_parent_avg", 0)
        ax.scatter(x, y, s=140, edgecolor="black")
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    lo = (
        min(
            min(m.get("cosine_to_base", 1) for m in metrics.values()),
            min(m.get("cosine_to_parent_avg", 1) for m in metrics.values()),
        )
        - 0.005
    )
    hi = 1.001
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("cosine(merged, base)")
    ax.set_ylabel("cosine(merged, parent average)")
    ax.set_title("Where each method lands in cosine-space")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 4. Top-5 most-changed layers per method (grouped bar)
def plot_top_layers_grouped(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    layer_drifts = _read(results_dir / "layer_drifts.json")
    if not layer_drifts:
        out.write_bytes(b"")
        return out
    methods = sorted(layer_drifts.keys())
    # union of top-5 layers per method
    top_layers: list[str] = []
    for m in methods:
        ranked = sorted(layer_drifts[m].items(), key=lambda x: x[1], reverse=True)[:5]
        for k, _ in ranked:
            if k not in top_layers:
                top_layers.append(k)
    if not top_layers:
        out.write_bytes(b"")
        return out
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(top_layers))
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(top_layers) + 4), 5))
    for i, m in enumerate(methods):
        vals = [layer_drifts[m].get(layer, 0) for layer in top_layers]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(top_layers, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("L2 drift")
    ax.set_title("Top-changed layers per method (grouped)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 5. mean_abs_drift comparison bar
def plot_drift_bar(results_dir: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for f in sorted(results_dir.glob("*__metrics.json")):
        m = f.stem.removesuffix("__metrics")
        metrics[m] = _read(f)
    if not metrics:
        out.write_bytes(b"")
        return out
    methods = list(metrics)
    vals = [metrics[m].get("mean_abs_drift", 0) for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(methods) + 2), 4.5))
    colors = matplotlib.colormaps["viridis"](np.linspace(0.2, 0.8, len(methods)))
    bars = ax.bar(methods, vals, color=colors)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{v:.4f}",
            ha="center",
            fontsize=8,
        )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("mean abs(merged - base)")
    ax.set_title("Per-method mean absolute drift from base")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
