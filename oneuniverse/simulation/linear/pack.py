"""Derive a `packed_npy` native dataset from a linear native dir.

Layout (one directory):

    {out}/
      header.json                  # box/grid/cosmology + per-product block map
      snapshots_z0.000.npy         # (N,6) particles, CHUNK-SORTED (x,y,z,vx,vy,vz)
      fields_z0.000.npy            # (n,n,n) density field
      halos_z0.000.parquet         # halos (small; left as parquet)
      lightcone.parquet            # sky (small; left as parquet)

The particle slab is sorted by cartesian chunk id so each chunk is a
contiguous [row_start, row_stop) range — the precondition for index-only
wrapping (Phase S17 T5). This mirrors how AbacusSummit / Gadget ship cells in
a spatial order. `PART_COLS` is the canonical particle column order.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from oneuniverse.simulation.oufsim.index import cartesian_chunk_ids, chunk_coords

PART_COLS = ("x", "y", "z", "vx", "vy", "vz")


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def write_packed_native(linear_dir: Union[str, Path], out_dir: Union[str, Path],
                        *, particle_chunk_nside: int = 4) -> Path:
    """Convert a linear native dir into a `packed_npy` native dir. Returns it."""
    linear_dir = Path(linear_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((linear_dir / "config.yaml").read_text())
    box = float(cfg["box_size"])
    n_side = int(particle_chunk_nside)

    header = {
        "native_format": "packed_npy",
        "box_size": box,
        "n_grid": int(cfg["n_grid"]),
        "redshifts": [float(z) for z in cfg["redshifts"]],
        "cosmology": cfg["cosmology"],
        "part_cols": list(PART_COLS),
        "snapshots": {}, "fields": {}, "halos": {}, "lightcone": None,
    }

    for z in cfg["redshifts"]:
        zt = _ztag(float(z))
        parts = np.load(linear_dir / zt / "particles.npy")          # (N,6)
        pos = parts[:, :3]
        cid = cartesian_chunk_ids(pos, box, n_side)
        order = np.argsort(cid, kind="stable")                       # chunk-sort
        parts_sorted = np.ascontiguousarray(parts[order])
        cid_sorted = cid[order]
        fname = f"snapshots_{zt}.npy"
        np.save(out / fname, parts_sorted)

        uniq, starts = np.unique(cid_sorted, return_index=True)
        bounds = list(starts) + [len(cid_sorted)]
        chunk_index = []
        for i, cc in enumerate(uniq):
            sl = slice(int(bounds[i]), int(bounds[i + 1]))
            p = parts_sorted[sl, :3]
            cx, cy, cz = chunk_coords(int(cc), n_side)
            chunk_index.append({
                "chunk_id": int(cc), "cx": cx, "cy": cy, "cz": cz,
                "row_start": int(sl.start), "row_stop": int(sl.stop),
                "n_rows": int(sl.stop - sl.start),
                "xlo": float(p[:, 0].min()), "xhi": float(p[:, 0].max()),
                "ylo": float(p[:, 1].min()), "yhi": float(p[:, 1].max()),
                "zlo": float(p[:, 2].min()), "zhi": float(p[:, 2].max()),
            })
        header["snapshots"][zt] = {"file": fname, "n_side": n_side,
                                   "chunk_index": chunk_index}

        ffname = f"fields_{zt}.npy"
        np.save(out / ffname, np.load(linear_dir / zt / "field.npy"))
        header["fields"][zt] = {"file": ffname}

        hsrc = linear_dir / zt / "halos.parquet"
        if hsrc.is_file():
            shutil.copy(hsrc, out / f"halos_{zt}.parquet")
            header["halos"][zt] = {"file": f"halos_{zt}.parquet"}

    lc = linear_dir / "lightcone.parquet"
    if lc.is_file():
        shutil.copy(lc, out / "lightcone.parquet")
        header["lightcone"] = {"file": "lightcone.parquet"}

    (out / "header.json").write_text(json.dumps(header, indent=2))
    return out
