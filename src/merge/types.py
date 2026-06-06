"""Core types.

A merge operates on a dict[str, Tensor] (the standard HF state_dict). We
keep everything as numpy arrays in the public API so the merge methods are
trivially testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

StateDict = dict[str, NDArray[np.float64]]


@dataclass
class MergeResult:
    method: str
    merged: StateDict
    extras: dict[str, float]
