"""Chunk, tile and sky index helpers, including periodic splitting of cubes.

Conventions
-----------
* Catalog chunks tile the periodic box in an ``n_side**3`` grid; the chunk id is
  ``(ix * n_side + iy) * n_side + iz`` (int32).  Positions are taken modulo the
  box, so ``x == box_size`` falls in chunk 0 and a tiny negative ``x`` in the
  last chunk.
* Field tiles are index ranges over an ``n_grid**3`` grid: ``i0..i1`` is
  half-open in cells, and cell ``i`` covers ``[i * cell, (i + 1) * cell)`` with
  ``cell = box_size / n_grid``.  A closed cube edge lying exactly on a cell
  boundary therefore touches the cell above it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube

CATALOG_INDEX_SCHEMA = pa.schema(
    [("chunk_id", pa.int32()), ("path", pa.string()),
     ("row_start", pa.int64()), ("row_stop", pa.int64())]
    + [(f"{a}_{m}", pa.float64()) for a in "xyz" for m in ("min", "max")])
FIELD_INDEX_SCHEMA = pa.schema(
    [("tile_id", pa.int32()), ("path", pa.string())]
    + [(f"{a}{e}", pa.int32()) for a in "ijk" for e in "01"])
SKY_INDEX_SCHEMA = pa.schema(
    [("pix", pa.int32()), ("path", pa.string()), ("n_rows", pa.int64())])


def _check_box(box_size: float, n: int, what: str) -> float:
    box = float(box_size)
    if not box > 0 or box == float("inf"):
        raise ValidationError(f"box_size must be finite and > 0, got {box_size}")
    if int(n) != n or n < 1:
        raise ValidationError(f"{what} must be a positive integer, got {n}")
    return box


def chunk_ids(pos: np.ndarray, box_size: float, n_side: int) -> np.ndarray:
    """Chunk id of each row of ``pos`` (n, 3); positions are wrapped into the box."""
    box = _check_box(box_size, n_side, "n_side")
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValidationError(f"pos must have shape (n, 3), got {pos.shape}")
    cell = np.floor(np.mod(pos, box) / (box / n_side)).astype(np.int64)
    np.clip(cell, 0, n_side - 1, out=cell)
    return ((cell[:, 0] * n_side + cell[:, 1]) * n_side + cell[:, 2]).astype(np.int32)


def chunk_bbox(chunk_id: int, box_size: float, n_side: int):
    """Inverse of :func:`chunk_ids`: ``(lo, hi)`` corner tuples of the chunk."""
    box = _check_box(box_size, n_side, "n_side")
    cid = int(chunk_id)
    if not 0 <= cid < n_side ** 3:
        raise ValidationError(f"chunk_id {chunk_id} outside [0, {n_side ** 3})")
    ix, rem = divmod(cid, n_side * n_side)
    iy, iz = divmod(rem, n_side)
    lo = tuple(i * box / n_side for i in (ix, iy, iz))
    hi = tuple((i + 1) * box / n_side for i in (ix, iy, iz))
    return lo, hi


def split_periodic(cube: Cube, box_size: float) -> list[Cube]:
    """Cut ``cube`` at the periodic box faces into 1..8 cubes inside ``[0, box_size]``.

    The cube is shifted so its lower corner lies in ``[0, box)`` and every axis
    that crosses ``box_size`` is split in two.  Order is deterministic (x
    outermost, unwrapped part before the wrapped part).  Volumes sum to the
    volume of ``cube``.
    """
    box = _check_box(box_size, 1, "n_side")
    spans = []
    for axis, (lo, hi) in enumerate(zip(cube.lo, cube.hi)):
        if hi - lo > box * (1 + 1e-12):
            raise ValidationError(
                f"cube side {hi - lo} on axis {axis} exceeds box_size {box}")
        if hi - lo > box:  # within rounding tolerance: clamp to one full box
            hi = lo + box
        shift = np.mod(lo, box) - lo
        if shift == 0.0:
            a, b = lo, hi
        else:
            a, b = lo + shift, hi + shift
        spans.append([(a, b)] if b <= box else [(a, box), (0.0, b - box)])
    return [Cube(tuple(s[0] for s in combo), tuple(s[1] for s in combo))
            for combo in itertools.product(*spans)]


def overlapping_chunks(index: pa.Table, cube: Cube, box_size: float) -> np.ndarray:
    """Sorted unique chunk ids whose closed bbox intersects ``cube`` (periodic)."""
    cid = index.column("chunk_id").to_numpy()
    lo = np.stack([index.column(f"{a}_min").to_numpy() for a in "xyz"], axis=1)
    hi = np.stack([index.column(f"{a}_max").to_numpy() for a in "xyz"], axis=1)
    keep = np.zeros(len(cid), dtype=bool)
    for part in split_periodic(cube, box_size):
        keep |= ((lo <= np.asarray(part.hi)) & (hi >= np.asarray(part.lo))).all(axis=1)
    return np.unique(cid[keep]).astype(np.int32)


def overlapping_tiles(index: pa.Table, cube: Cube, box_size: float, n_grid: int) -> np.ndarray:
    """Sorted unique tile ids whose cell ranges intersect ``cube`` (periodic)."""
    box = _check_box(box_size, n_grid, "n_grid")
    cell = box / n_grid
    tid = index.column("tile_id").to_numpy()
    start = np.stack([index.column(f"{a}0").to_numpy() for a in "ijk"], axis=1)
    stop = np.stack([index.column(f"{a}1").to_numpy() for a in "ijk"], axis=1)
    keep = np.zeros(len(tid), dtype=bool)
    for part in split_periodic(cube, box_size):
        first = np.clip(np.floor(np.asarray(part.lo) / cell), 0, n_grid - 1).astype(np.int64)
        last = np.clip(np.floor(np.asarray(part.hi) / cell), 0, n_grid - 1).astype(np.int64)
        keep |= ((start <= last) & (stop > first)).all(axis=1)
    return np.unique(tid[keep]).astype(np.int32)
