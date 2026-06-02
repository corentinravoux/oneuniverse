"""SimStore — partial-access reader over an OUF-Sim store.

The load-bearing Pillar-3 idea: a selector (Cube / Cone / SkyPatch) is
resolved against each product's sidecar ``_index.parquet`` so that only
the overlapping parquet chunks (or memmap field tiles, or HEALPix
super-pixels) are touched — never the whole snapshot. ``last_read_stats``
records how many pieces were considered vs actually read, which is the
quantity Phase-S5/optimisation work tunes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.oufsim._io import read_json
from oneuniverse.simulation.oufsim.index import (
    cone_partition_pixels,
    cube_overlaps_bbox,
)
from oneuniverse.simulation.selectors import Cone, Cube


class SimStore:
    """Open an OUF-Sim store directory (the one holding ``manifest.json``)."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        self.layout = self.manifest.get("store_layout", {})
        self.last_read_stats: Dict[str, int] = {}

    # -- introspection ----------------------------------------------------
    @property
    def products(self):
        return tuple(self.manifest.get("products", ()))

    @property
    def redshifts(self):
        return tuple(self.manifest.get("redshifts", ()))

    def _index_rows(self, rel_index: str):
        return pq.read_table(self.root / rel_index).to_pylist()

    # -- point catalogues (particles / halos) -----------------------------
    def read_box(self, product: str, z: float, cube: Cube) -> Dict[str, np.ndarray]:
        """Return columns inside ``cube`` for a point product at redshift z."""
        zt = f"z{float(z):.3f}"
        info = self.layout[product][zt]
        rows = self._index_rows(info["index"])
        prod_dir = self.root / info["dir"]
        hit = [r for r in rows
               if cube_overlaps_bbox(cube, (r["xlo"], r["xhi"], r["ylo"],
                                            r["yhi"], r["zlo"], r["zhi"]))]
        self.last_read_stats = {"chunks_total": len(rows), "chunks_read": len(hit)}
        cols: Dict[str, list] = {}
        for r in hit:
            table = pq.read_table(prod_dir / r["file"])
            for name in table.column_names:
                cols.setdefault(name, []).append(
                    table.column(name).to_numpy(zero_copy_only=False))
        if not cols:
            return {}
        out = {k: np.concatenate(v) for k, v in cols.items()}
        m = ((out["x"] >= cube.xlo) & (out["x"] <= cube.xhi)
             & (out["y"] >= cube.ylo) & (out["y"] <= cube.yhi)
             & (out["z"] >= cube.zlo) & (out["z"] <= cube.zhi))
        return {k: v[m] for k, v in out.items()}

    # -- field (regular grid) ---------------------------------------------
    def read_field_box(self, z: float, cube: Cube):
        """Stitch the field sub-grid covering ``cube`` from memmap tiles.

        Returns ``(subgrid, origin_cell)`` where origin_cell is the
        (ix0, iy0, iz0) cell index of the sub-grid's corner.
        """
        zt = f"z{float(z):.3f}"
        info = self.layout["fields"][zt]
        rows = self._index_rows(info["index"])
        prod_dir = self.root / info["dir"]
        n_grid = int(self.manifest["n_grid"])
        box = float(self.manifest["box_size"])
        cell = box / n_grid

        ix0 = max(0, int(math.floor(cube.xlo / cell)))
        ix1 = min(n_grid, int(math.ceil(cube.xhi / cell)))
        iy0 = max(0, int(math.floor(cube.ylo / cell)))
        iy1 = min(n_grid, int(math.ceil(cube.yhi / cell)))
        iz0 = max(0, int(math.floor(cube.zlo / cell)))
        iz1 = min(n_grid, int(math.ceil(cube.zhi / cell)))
        sub = np.zeros((ix1 - ix0, iy1 - iy0, iz1 - iz0), dtype=np.float64)

        n_read = 0
        for r in rows:
            ax0, ax1 = max(ix0, r["ix0"]), min(ix1, r["ix1"])
            ay0, ay1 = max(iy0, r["iy0"]), min(iy1, r["iy1"])
            az0, az1 = max(iz0, r["iz0"]), min(iz1, r["iz1"])
            if ax0 >= ax1 or ay0 >= ay1 or az0 >= az1:
                continue
            tile = np.load(prod_dir / r["file"], mmap_mode="r")
            sub[ax0 - ix0:ax1 - ix0, ay0 - iy0:ay1 - iy0, az0 - iz0:az1 - iz0] = \
                tile[ax0 - r["ix0"]:ax1 - r["ix0"],
                     ay0 - r["iy0"]:ay1 - r["iy0"],
                     az0 - r["iz0"]:az1 - r["iz0"]]
            n_read += 1
        self.last_read_stats = {"tiles_total": len(rows), "tiles_read": n_read}
        return sub, (ix0, iy0, iz0)

    # -- lightcone (sky) --------------------------------------------------
    def read_cone(self, cone: Cone) -> Dict[str, np.ndarray]:
        """Return lightcone objects within ``cone`` (HEALPix-pruned)."""
        info = self.layout["lightcone"]
        rows = self._index_rows(info["index"])
        prod_dir = self.root / info["dir"]
        nside_part = int(info["nside_part"])
        want = set(int(p) for p in cone_partition_pixels(cone, nside_part))
        hit = [r for r in rows if int(r["pixel"]) in want]
        self.last_read_stats = {"pixels_total": len(rows), "pixels_read": len(hit)}
        cols: Dict[str, list] = {}
        for r in hit:
            table = pq.read_table(prod_dir / r["file"])
            for name in table.column_names:
                cols.setdefault(name, []).append(
                    table.column(name).to_numpy(zero_copy_only=False))
        if not cols:
            return {}
        out = {k: np.concatenate(v) for k, v in cols.items()}
        # precise angular cut
        dlon = np.radians(out["lon"] - cone.lon)
        lat1 = np.radians(out["lat"])
        lat0 = np.radians(cone.lat)
        cosang = (np.sin(lat0) * np.sin(lat1)
                  + np.cos(lat0) * np.cos(lat1) * np.cos(dlon))
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        m = ang <= cone.radius_deg
        return {k: v[m] for k, v in out.items()}
