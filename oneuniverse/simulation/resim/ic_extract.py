"""Extract a Lagrangian sub-region IC from a parent field.

The mini-sim IC is a sub-cube of the parent field (white noise or density),
so it inherits the parent's phases + the large-scale modes present within
the cube exactly. This is the phase-consistent zoom-IC step (the strong,
reliable part of the feasibility analysis).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from oneuniverse.simulation.selectors import Cube


def extract_region(field: np.ndarray, cube: Cube, *, box_size: float
                   ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Return (sub-grid, origin_cell) of ``field`` covering ``cube``."""
    n = field.shape[0]
    cell = box_size / n
    ix0 = max(0, int(math.floor(cube.xlo / cell)))
    ix1 = min(n, int(math.ceil(cube.xhi / cell)))
    iy0 = max(0, int(math.floor(cube.ylo / cell)))
    iy1 = min(n, int(math.ceil(cube.yhi / cell)))
    iz0 = max(0, int(math.floor(cube.zlo / cell)))
    iz1 = min(n, int(math.ceil(cube.zhi / cell)))
    sub = np.ascontiguousarray(field[ix0:ix1, iy0:iy1, iz0:iz1])
    return sub, (ix0, iy0, iz0)
