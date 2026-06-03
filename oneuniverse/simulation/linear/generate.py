"""Top-level dummy-simulation writer.

Generates the field (mesh/voxel), Zel'dovich particles, and toy halos
for a list of redshifts and writes a simple native on-disk layout that
the Phase-S4 LinearSimConverter will wrap into OUF-Sim:

    {out_dir}/
    |- config.yaml                 (cosmology + box + grid + seed + redshifts)
    |- z0.000/
    |   |- field.npy               (n,n,n) float64 density contrast
    |   |- particles.npy           (n^3, 6) float64 x,y,z,vx,vy,vz
    |   `- halos.parquet           toy halo catalogue
    `- z0.500/ ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.amr import refine_field
from oneuniverse.simulation.linear.gr_fields import potential_field
from oneuniverse.simulation.linear.ic import white_noise_ic
from oneuniverse.simulation.linear.lightcone import build_lightcone_catalog
from oneuniverse.simulation.linear.phase_space import phase_space_sheet
from oneuniverse.simulation.linear.tree import build_merger_tree
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def generate_linear_sim(
    out_dir: Union[str, Path],
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    redshifts: Sequence[float],
    seed: int = 0,
    halo_threshold: float = 1.0,
    with_lightcone: bool = True,
) -> Path:
    """Generate + write a dummy linear simulation. Returns the root dir."""
    c = require_cosmo(cosmo)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "generator": "oneuniverse.simulation.linear",
        "box_size": float(box_size),
        "n_grid": int(n_grid),
        "redshifts": [float(z) for z in redshifts],
        "seed": int(seed),
        "halo_threshold": float(halo_threshold),
        "with_lightcone": bool(with_lightcone),
        "cosmology": c.to_dict(),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    fields_by_z = {}
    halos_by_z = {}
    for z in redshifts:
        zdir = out / _ztag(z)
        zdir.mkdir(parents=True, exist_ok=True)

        field = generate_density_field(
            c, box_size=box_size, n_grid=n_grid, z=z, seed=seed,
        )
        np.save(zdir / "field.npy", field)
        fields_by_z[float(z)] = field

        pos, vel = zeldovich_particles(
            c, box_size=box_size, n_grid=n_grid, z=z, seed=seed,
        )
        parts = np.concatenate([pos, vel], axis=1)  # (n^3, 6)
        np.save(zdir / "particles.npy", parts)

        halos = find_peaks(field, box_size=box_size, threshold=halo_threshold)
        halos_by_z[float(z)] = halos
        table = pa.table({k: pa.array(v) for k, v in halos.items()})
        pq.write_table(table, zdir / "halos.parquet")

        ps = phase_space_sheet(c, box_size=box_size, n_grid=n_grid, z=z,
                               seed=seed)
        pq.write_table(pa.table({k: pa.array(v) for k, v in ps.items()}),
                       zdir / "phase_space.parquet")
        np.save(zdir / "gr_field.npy", potential_field(field, box_size=box_size))

        amr = refine_field(field, threshold=halo_threshold)
        amr_cols = {k: amr[k] for k in ("parent_ix", "parent_iy", "parent_iz",
                                        "node_id")}
        amr_cols["node_id"] = amr_cols["node_id"].astype(np.int64)
        for j in range(8):
            amr_cols[f"sub{j}"] = (amr["subcells"][:, j] if amr["n_refined"]
                                   else np.empty(0))
        pq.write_table(pa.table({k: pa.array(v) for k, v in amr_cols.items()}),
                       zdir / "amr.parquet")

    # Initial-conditions (input) product — the white-noise realisation.
    ic_field, ic_desc = white_noise_ic(c, box_size=box_size, n_grid=n_grid,
                                       seed=seed)
    np.save(out / "ic_field.npy", ic_field)
    (out / "ic_descriptor.json").write_text(json.dumps(ic_desc, indent=2))

    # Reproducible IC / checkpoint descriptor (the deterministic IC; no run).
    (out / "checkpoint.json").write_text(json.dumps({
        "seed": int(seed), "box_size": float(box_size), "n_grid": int(n_grid),
        "redshifts": [float(z) for z in redshifts], "cosmology": c.to_dict(),
    }, indent=2))

    if len(redshifts) >= 2:
        tree = build_merger_tree(halos_by_z, box_size=box_size)
        tree_table = pa.table({k: pa.array(v) for k, v in tree.items()})
        pq.write_table(tree_table, out / "tree.parquet")

    if with_lightcone:
        lc = build_lightcone_catalog(
            fields_by_z, box_size=box_size, halo_threshold=halo_threshold,
        )
        lc_table = pa.table({k: pa.array(v) for k, v in lc.items()})
        pq.write_table(lc_table, out / "lightcone.parquet")

    return out
