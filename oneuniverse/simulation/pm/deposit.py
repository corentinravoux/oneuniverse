"""Cloud-in-cell (CIC) deposit + interpolation (periodic)."""
from __future__ import annotations

import numpy as np


def _cic_nodes(pos, n_grid, box):
    cell = box / n_grid
    g = (np.asarray(pos, dtype=np.float64) / cell) % n_grid
    i0 = np.floor(g).astype(np.int64) % n_grid
    d = g - np.floor(g)
    i1 = (i0 + 1) % n_grid
    return i0, i1, d


def deposit_cic(pos: np.ndarray, n_grid: int, box: float) -> np.ndarray:
    """Deposit unit-mass particles onto an n_grid^3 mesh (mass per cell)."""
    i0, i1, d = _cic_nodes(pos, n_grid, box)
    rho = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    ix = (i0[:, 0], i1[:, 0]); iy = (i0[:, 1], i1[:, 1]); iz = (i0[:, 2], i1[:, 2])
    wx = (1.0 - d[:, 0], d[:, 0])
    wy = (1.0 - d[:, 1], d[:, 1])
    wz = (1.0 - d[:, 2], d[:, 2])
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                np.add.at(rho, (ix[a], iy[b], iz[c]), wx[a] * wy[b] * wz[c])
    return rho


def interpolate_cic(grid: np.ndarray, pos: np.ndarray, box: float) -> np.ndarray:
    """Sample a mesh at particle positions with CIC weights."""
    n_grid = grid.shape[0]
    i0, i1, d = _cic_nodes(pos, n_grid, box)
    ix = (i0[:, 0], i1[:, 0]); iy = (i0[:, 1], i1[:, 1]); iz = (i0[:, 2], i1[:, 2])
    wx = (1.0 - d[:, 0], d[:, 0])
    wy = (1.0 - d[:, 1], d[:, 1])
    wz = (1.0 - d[:, 2], d[:, 2])
    out = np.zeros(len(pos), dtype=np.float64)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                out += grid[ix[a], iy[b], iz[c]] * wx[a] * wy[b] * wz[c]
    return out
