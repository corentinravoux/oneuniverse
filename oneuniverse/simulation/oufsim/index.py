"""Layer-1 index toolkit — format-agnostic spatial / sky partitioning.

These primitives are what a converter's Layer-1 step calls to make a
product partial-access-able: assign rows/cells to chunks, compute
bounding boxes, and test a selector against the sidecar index. They know
nothing about any simulation code — only numpy arrays + selectors.
"""
from __future__ import annotations

from typing import List, Tuple

import healpy as hp
import numpy as np

from oneuniverse.simulation.selectors import Cone, Cube, SkyPatch


# --------------------------------------------------------------------------
# Cartesian point chunking (particles, halos)
# --------------------------------------------------------------------------
def cartesian_chunk_ids(pos: np.ndarray, box_size: float, n_side: int) -> np.ndarray:
    """Assign each point to a coarse cube chunk; returns int chunk_id (N,)."""
    cell = box_size / n_side
    c = np.clip((pos // cell).astype(np.int64), 0, n_side - 1)
    return (c[:, 0] * n_side + c[:, 1]) * n_side + c[:, 2]


def chunk_coords(chunk_id: int, n_side: int) -> Tuple[int, int, int]:
    cz = chunk_id % n_side
    cy = (chunk_id // n_side) % n_side
    cx = chunk_id // (n_side * n_side)
    return int(cx), int(cy), int(cz)


def bbox_of(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Axis-aligned bounding box (xlo,xhi,ylo,yhi,zlo,zhi) of (N,3) points."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]),
            float(lo[2]), float(hi[2]))


def cube_overlaps_bbox(cube: Cube, bbox) -> bool:
    """True if the selector cube overlaps an axis-aligned bbox."""
    xlo, xhi, ylo, yhi, zlo, zhi = bbox
    return not (
        cube.xhi < xlo or cube.xlo > xhi
        or cube.yhi < ylo or cube.ylo > yhi
        or cube.zhi < zlo or cube.zlo > zhi
    )


# --------------------------------------------------------------------------
# Regular-grid tiling (fields / voxels)
# --------------------------------------------------------------------------
def tile_specs(n_grid: int, tile_cells: int) -> List[dict]:
    """Split an n_grid^3 mesh into cubic tiles of ``tile_cells`` per side.

    Returns dicts with cell-index ranges (half-open) per tile.
    """
    edges = list(range(0, n_grid, tile_cells)) + [n_grid]
    bounds = [(edges[i], min(edges[i] + tile_cells, n_grid))
              for i in range(len(edges) - 1) if edges[i] < n_grid]
    specs = []
    tid = 0
    for ix0, ix1 in bounds:
        for iy0, iy1 in bounds:
            for iz0, iz1 in bounds:
                specs.append({
                    "tile_id": tid,
                    "ix0": ix0, "ix1": ix1,
                    "iy0": iy0, "iy1": iy1,
                    "iz0": iz0, "iz1": iz1,
                })
                tid += 1
    return specs


def tile_overlaps_cube(spec: dict, cube: Cube, box_size: float, n_grid: int) -> bool:
    """True if a field tile (cell-index ranges) overlaps the selector cube."""
    cell = box_size / n_grid
    bbox = (
        spec["ix0"] * cell, spec["ix1"] * cell,
        spec["iy0"] * cell, spec["iy1"] * cell,
        spec["iz0"] * cell, spec["iz1"] * cell,
    )
    return cube_overlaps_bbox(cube, bbox)


# --------------------------------------------------------------------------
# Sky partitioning (lightcone) — HEALPix NEST, like OUF
# --------------------------------------------------------------------------
def healpix_partition_ids(lon_deg, lat_deg, nside_part: int) -> np.ndarray:
    """NEST super-pixel id for each (lon, lat) at ``nside_part``."""
    theta = np.radians(90.0 - np.asarray(lat_deg, dtype=np.float64))
    phi = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return hp.ang2pix(nside_part, theta, phi, nest=True).astype(np.int64)


def cone_partition_pixels(cone: Cone, nside_part: int) -> np.ndarray:
    """Super-pixels (NEST) that a Cone disc can touch — superset, safe."""
    theta = np.radians(90.0 - cone.lat)
    phi = np.radians(cone.lon)
    vec = hp.ang2vec(theta, phi)
    radius = np.radians(cone.radius_deg)
    # inclusive=True returns a superset so no overlapping pixel is missed.
    return hp.query_disc(nside_part, vec, radius, nest=True, inclusive=True)


def skypatch_partition_pixels(patch: SkyPatch, nside_part: int) -> np.ndarray:
    """Super-pixels (NEST) covering a lon/lat rectangle (corner polygon)."""
    lons = [patch.lon_min, patch.lon_max, patch.lon_max, patch.lon_min]
    lats = [patch.lat_min, patch.lat_min, patch.lat_max, patch.lat_max]
    theta = np.radians(90.0 - np.asarray(lats))
    phi = np.radians(np.asarray(lons))
    vecs = hp.ang2vec(theta, phi)
    return hp.query_polygon(nside_part, vecs, nest=True, inclusive=True)
