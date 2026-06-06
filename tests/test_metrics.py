from __future__ import annotations

import numpy as np

from merge.metrics.compare import cosine, flat, l2_per_layer, mean_abs_drift, top_k_changes


def _mk(seed: int, n_layers: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    return {f"l{i}": rng.standard_normal((4, 4)).astype(np.float64) for i in range(n_layers)}


def test_flat_concatenates() -> None:
    s = _mk(0, n_layers=2)
    f = flat(s)
    assert f.size == 32


def test_cosine_self_is_one() -> None:
    s = _mk(0)
    assert abs(cosine(s, s) - 1.0) < 1e-9


def test_cosine_zero_state_is_zero() -> None:
    a = {"l0": np.zeros((4, 4))}
    b = {"l0": np.ones((4, 4))}
    assert cosine(a, b) == 0.0


def test_l2_per_layer_returns_dict() -> None:
    a = _mk(0)
    b = _mk(1)
    d = l2_per_layer(a, b)
    assert len(d) == 3
    assert all(v >= 0 for v in d.values())


def test_mean_abs_drift_zero_when_identical() -> None:
    a = _mk(0)
    assert mean_abs_drift(a, a) == 0.0


def test_top_k_changes_returns_sorted_desc() -> None:
    a = _mk(0)
    b = _mk(1)
    top = top_k_changes(a, b, k=2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]
