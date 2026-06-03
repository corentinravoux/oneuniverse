"""Merge resimulated sub-regions into a global field.

Each resimulated tile is placed on the global grid weighted by a feather
window (linear edge ramp → smooth blend across overlaps, no seam). The merged
value is the weighted average, so where tiles agree they reproduce the field
exactly, and where they disagree (different resimulations) the overlap blends
smoothly. Periodic placement (wraps at the box edge).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def feather_window(shape: Sequence[int], feather: int) -> np.ndarray:
    """1 in the interior, linear ramp 0→1 over ``feather`` cells at each edge."""
    w = np.ones(shape, dtype=np.float64)
    if feather <= 0:
        return w
    for ax, nax in enumerate(shape):
        i = np.arange(nax)
        d = np.minimum(i, nax - 1 - i)               # distance to nearest edge
        r = np.clip((d + 0.5) / feather, 0.0, 1.0)
        sh = [1] * len(shape); sh[ax] = nax
        w = w * r.reshape(sh)
    return w


def merge_fields(global_shape: Tuple[int, int, int],
                 tiles: List[dict], *, feather: int = 0) -> np.ndarray:
    """Merge tiles `[{field, origin}]` into a `global_shape` grid (periodic)."""
    acc = np.zeros(global_shape, dtype=np.float64)
    wsum = np.zeros(global_shape, dtype=np.float64)
    for t in tiles:
        f = np.asarray(t["field"], dtype=np.float64)
        o = t["origin"]
        w = feather_window(f.shape, feather)
        ix = [(o[a] + np.arange(f.shape[a])) % global_shape[a] for a in range(3)]
        gx = np.ix_(ix[0], ix[1], ix[2])
        acc[gx] += w * f
        wsum[gx] += w
    return acc / np.maximum(wsum, 1e-12)
