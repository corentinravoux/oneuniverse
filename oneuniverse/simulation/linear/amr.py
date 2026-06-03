"""Toy 1-level AMR refinement around density peaks.

Refines every base cell with delta > threshold into an octant of 8
sub-cells (one level, 2x), giving the non-Cartesian ``amr`` field layout.
Sub-cell values = parent delta + a half-cell trilinear perturbation from the
periodic neighbour gradients. Each refined node carries a Morton key for the
octree-node index. Pure numpy.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

_OCTANTS = np.array([(sx, sy, sz)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                    dtype=np.float64)  # (8, 3)


def _morton3(ix, iy, iz) -> np.ndarray:
    def part(n):
        n = n.astype(np.uint64) & np.uint64(0x3ff)
        n = (n | (n << np.uint64(16))) & np.uint64(0x30000ff)
        n = (n | (n << np.uint64(8))) & np.uint64(0x300f00f)
        n = (n | (n << np.uint64(4))) & np.uint64(0x30c30c3)
        n = (n | (n << np.uint64(2))) & np.uint64(0x9249249)
        return n
    return (part(ix) | (part(iy) << np.uint64(1)) | (part(iz) << np.uint64(2)))


def refine_field(delta: np.ndarray, *, threshold: float = 1.0,
                 level: int = 1) -> Dict[str, np.ndarray]:
    """Refine peak cells into 8 sub-cells each; return the refined-node table."""
    d = np.asarray(delta, dtype=np.float64)
    mask = d > threshold
    idx = np.argwhere(mask)               # (N, 3) refined parent cells
    if idx.size == 0:
        e = np.empty(0, dtype=np.int64)
        return {"parent_ix": e, "parent_iy": e.copy(), "parent_iz": e.copy(),
                "node_id": np.empty(0, dtype=np.uint64),
                "subcells": np.empty((0, 8), dtype=np.float64),
                "n_refined": 0, "level": int(level)}

    # periodic central-difference gradients
    gx = (np.roll(d, -1, 0) - np.roll(d, 1, 0)) * 0.5
    gy = (np.roll(d, -1, 1) - np.roll(d, 1, 1)) * 0.5
    gz = (np.roll(d, -1, 2) - np.roll(d, 1, 2)) * 0.5
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
    parent = d[ix, iy, iz][:, None]                       # (N, 1)
    grad = np.stack([gx[ix, iy, iz], gy[ix, iy, iz],
                     gz[ix, iy, iz]], axis=1)             # (N, 3)
    # subcell value per octant: parent + 0.25 * (octant . grad)
    subcells = parent + 0.25 * (grad @ _OCTANTS.T)        # (N, 8)
    return {
        "parent_ix": ix.astype(np.int64),
        "parent_iy": iy.astype(np.int64),
        "parent_iz": iz.astype(np.int64),
        "node_id": _morton3(ix, iy, iz),
        "subcells": subcells,
        "n_refined": int(idx.shape[0]),
        "level": int(level),
    }
