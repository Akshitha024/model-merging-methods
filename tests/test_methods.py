from __future__ import annotations

import numpy as np
import pytest

from merge.methods.methods import dare, linear, model_stock, slerp, task_arithmetic, ties


def _mk(seed: int, shape: tuple[int, ...] = (4, 4)) -> dict:
    rng = np.random.default_rng(seed)
    return {"layer_0.weight": rng.standard_normal(shape).astype(np.float64)}


def test_linear_equal_weights_is_average() -> None:
    a = _mk(1)
    b = _mk(2)
    m = linear([a, b])
    expected = 0.5 * a["layer_0.weight"] + 0.5 * b["layer_0.weight"]
    np.testing.assert_allclose(m["layer_0.weight"], expected, atol=1e-9)


def test_linear_requires_matching_weights() -> None:
    with pytest.raises(ValueError):
        linear([_mk(1), _mk(2)], weights=[0.1, 0.2, 0.7])


def test_slerp_midpoint_is_close_to_lerp_for_small_angle() -> None:
    a = _mk(1)
    b = _mk(2)
    s = slerp([a, b])
    midpoint = 0.5 * a["layer_0.weight"] + 0.5 * b["layer_0.weight"]
    # for small-angle random vectors slerp and lerp are similar; loose tolerance
    assert np.linalg.norm(s["layer_0.weight"] - midpoint) < np.linalg.norm(midpoint)


def test_task_arithmetic_zero_coefficient_returns_base() -> None:
    base = _mk(0)
    a = _mk(1)
    out = task_arithmetic(base, [a], coefficients=[0.0])
    np.testing.assert_allclose(out["layer_0.weight"], base["layer_0.weight"], atol=1e-12)


def test_task_arithmetic_unit_one_parent_returns_parent() -> None:
    base = _mk(0)
    a = _mk(1)
    out = task_arithmetic(base, [a], coefficients=[1.0])
    np.testing.assert_allclose(out["layer_0.weight"], a["layer_0.weight"], atol=1e-12)


def test_ties_runs_and_preserves_shape() -> None:
    base = _mk(0, shape=(8, 8))
    parents = [_mk(i, shape=(8, 8)) for i in range(1, 4)]
    out = ties(base, parents, k_percent=0.3)
    assert out["layer_0.weight"].shape == (8, 8)


def test_ties_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        ties(_mk(0), [_mk(1)], k_percent=0)
    with pytest.raises(ValueError):
        ties(_mk(0), [_mk(1)], k_percent=1.5)


def test_dare_with_zero_drop_equals_linear_task_sum() -> None:
    base = _mk(0)
    a = _mk(1)
    b = _mk(2)
    out = dare(base, [a, b], drop_p=0.0)
    # equal coefficients = (a+b)/2, applied as task vector on base
    expected = (
        base["layer_0.weight"]
        + 0.5 * (a["layer_0.weight"] - base["layer_0.weight"])
        + 0.5 * (b["layer_0.weight"] - base["layer_0.weight"])
    )
    np.testing.assert_allclose(out["layer_0.weight"], expected, atol=1e-9)


def test_model_stock_two_parents_is_midpoint() -> None:
    a = _mk(1)
    b = _mk(2)
    out = model_stock([a, b])
    midpoint = 0.5 * a["layer_0.weight"] + 0.5 * b["layer_0.weight"]
    np.testing.assert_allclose(out["layer_0.weight"], midpoint, atol=1e-9)


def test_model_stock_requires_two() -> None:
    with pytest.raises(ValueError):
        model_stock([_mk(1)])
