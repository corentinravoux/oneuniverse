"""Toy merger tree from the per-redshift halo catalogues.

Links halos across adjacent snapshots by nearest neighbour in comoving
(periodic) position: the lower-z halo is the *descendant*, its nearest
higher-z halo the *progenitor*. Produces an edge list — the OUF-Sim
``tree`` product. Pure numpy + (lazy) scipy KD-tree with a brute-force
fallback.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _nearest(tree_points: np.ndarray, query: np.ndarray,
             box_size: float) -> np.ndarray:
    """Index into ``tree_points`` of the nearest (periodic) point per query."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        # periodic brute force (fine for toy catalogues)
        idx = np.empty(len(query), dtype=np.int64)
        for i, q in enumerate(query):
            d = np.abs(tree_points - q)
            d = np.minimum(d, box_size - d)
            idx[i] = int(np.argmin((d ** 2).sum(axis=1)))
        return idx
    t = cKDTree(tree_points, boxsize=box_size)
    return t.query(query)[1].astype(np.int64)


def build_merger_tree(halos_by_z: Dict[float, Dict[str, np.ndarray]], *,
                      box_size: float) -> Dict[str, np.ndarray]:
    """Return a progenitor/descendant edge list across adjacent redshifts."""
    zs = sorted(halos_by_z)
    desc_id, prog_id, z_desc, z_prog = [], [], [], []
    for zd, zp in zip(zs[:-1], zs[1:]):           # zd < zp
        hd, hp = halos_by_z[zd], halos_by_z[zp]
        if len(hd["x"]) == 0 or len(hp["x"]) == 0:
            continue
        pd = np.stack([hd["x"], hd["y"], hd["z"]], axis=1) % box_size
        pp = np.stack([hp["x"], hp["y"], hp["z"]], axis=1) % box_size
        nn = _nearest(pp, pd, box_size)
        desc_id.append(np.asarray(hd["halo_id"], dtype=np.int64))
        prog_id.append(np.asarray(hp["halo_id"], dtype=np.int64)[nn])
        z_desc.append(np.full(len(pd), float(zd)))
        z_prog.append(np.full(len(pd), float(zp)))

    if not desc_id:
        e = np.empty(0, dtype=np.int64); f = np.empty(0, dtype=np.float64)
        return {"descendant_id": e, "progenitor_id": e.copy(),
                "z_desc": f, "z_prog": f.copy()}
    return {
        "descendant_id": np.concatenate(desc_id),
        "progenitor_id": np.concatenate(prog_id),
        "z_desc": np.concatenate(z_desc),
        "z_prog": np.concatenate(z_prog),
    }
