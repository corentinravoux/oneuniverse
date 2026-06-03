"""Morton (Z-order) keys for spatially clustering rows within a chunk.

Sorting a chunk's rows by Morton key makes consecutive rows spatially close,
so each parquet row-group spans a small bounding box → predicate pushdown
(row-group min/max stats) can skip most groups for a sub-cube query.
"""
from __future__ import annotations

import numpy as np

_MASKS = (
    np.uint64(0x1fffff),
    np.uint64(0x1f00000000ffff),
    np.uint64(0x1f0000ff0000ff),
    np.uint64(0x100f00f00f00f00f),
    np.uint64(0x10c30c30c30c30c3),
    np.uint64(0x1249249249249249),
)
_SHIFTS = (np.uint64(32), np.uint64(16), np.uint64(8), np.uint64(4), np.uint64(2))


def _part1by2(n: np.ndarray) -> np.ndarray:
    n = n & _MASKS[0]
    n = (n | (n << _SHIFTS[0])) & _MASKS[1]
    n = (n | (n << _SHIFTS[1])) & _MASKS[2]
    n = (n | (n << _SHIFTS[2])) & _MASKS[3]
    n = (n | (n << _SHIFTS[3])) & _MASKS[4]
    n = (n | (n << _SHIFTS[4])) & _MASKS[5]
    return n


def morton_key(pos: np.ndarray, box_size: float, bits: int = 20) -> np.ndarray:
    """21-bit-per-axis Z-order key for (N,3) positions in [0, box_size)."""
    scale = (1 << bits) / box_size
    q = np.clip((np.asarray(pos) * scale).astype(np.int64), 0,
                (1 << bits) - 1).astype(np.uint64)
    return (_part1by2(q[:, 0])
            | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))
