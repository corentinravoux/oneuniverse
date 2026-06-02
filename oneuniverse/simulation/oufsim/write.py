"""Write an OUF-Sim store from a native linear-sim layout.

Produces a store that mirrors OUF's tech stack:

    {out_root}/{sim_name}/oufsim/
      manifest.json                         (JSON, atomic, version-pinned)
      snapshots/z*/  part_*.parquet + _index.parquet   (Cube partial access)
      fields/z*/     tile_*.npy   + _index.parquet      (memmap tiles)
      halos/z*/      part_*.parquet + _index.parquet
      lightcone/     part_*.parquet + _index.parquet    (HEALPix NEST)

Each product carries a sidecar ``_index.parquet`` (per-chunk bbox or
HEALPix super-pixel) so a selector reads only the overlapping pieces.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from oneuniverse.simulation._version import OUFSIM_FORMAT_VERSION
from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.manifest import OUFSimManifest
from oneuniverse.simulation.oufsim._io import write_json
from oneuniverse.simulation.oufsim._parallel import map_partitions
from oneuniverse.simulation.oufsim.index import (
    cartesian_chunk_ids,
    chunk_coords,
    healpix_partition_ids,
    tile_specs,
)
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec

OUFSIM_SUBDIR = "oufsim"
INDEX_FILE = "_index.parquet"
_COMPRESSION = "snappy"  # same default as OUF

# Heavy write steps + the modes the OUF-Sim writer can actually deliver.
# A requested ExecutionPlan mode absent here is refused (Rule 5: never
# silently degrade to an unbounded in-memory path).
_WRITE_CAPS = BackendCapabilities(
    name="oufsim-writer",
    native_format="parquet + .npy tiles",
    supports_mpi=True,
    supports_streaming=True,
    heavy_step_modes={
        "particle_chunking": (ExecutionMode.SEQUENTIAL, ExecutionMode.MPI),
        "parquet_write": (ExecutionMode.SEQUENTIAL, ExecutionMode.MPI),
        "field_tiling": (ExecutionMode.SEQUENTIAL,),
    },
)
_HEAVY_STEPS = ("particle_chunking", "parquet_write", "field_tiling")


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def _write_index(path: Path, rows: list) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression=_COMPRESSION)


# --------------------------------------------------------------------------
# Point-catalogue products (particles, halos) — Cube-partitioned
# --------------------------------------------------------------------------
def _write_chunked_catalog(
    prod_dir: Path,
    columns: Dict[str, np.ndarray],
    pos: np.ndarray,
    box_size: float,
    n_side: int,
    batch_rows: Optional[int] = None,
    n_threads: int = 1,
    use_mpi: bool = False,
) -> dict:
    """Cube-chunk a point catalogue; one file per chunk, written in parallel.

    Streaming: no global sorted *copy* of every column (the S4 hotspot) — we
    keep only the int ``order`` index and gather one chunk's rows at a time;
    bounding boxes are fused into a single ``minimum.at``/``maximum.at`` pass.
    Per-chunk writes (the 52% hotspot) dispatch across threads / MPI ranks
    via ``map_partitions``; the index is reassembled deterministically.
    ``batch_rows`` is reserved for capping the per-rank working set.
    """
    prod_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = cartesian_chunk_ids(pos, box_size, n_side)
    n_chunks = n_side ** 3

    # Fused per-chunk bbox in one streaming pass (no per-chunk reductions).
    counts = np.bincount(chunk_ids, minlength=n_chunks)
    lo = np.full((n_chunks, 3), np.inf)
    hi = np.full((n_chunks, 3), -np.inf)
    np.minimum.at(lo, chunk_ids, pos)
    np.maximum.at(hi, chunk_ids, pos)

    order = np.argsort(chunk_ids, kind="stable")
    ids_sorted = chunk_ids[order]
    uniq, starts = np.unique(ids_sorted, return_index=True)
    starts = list(starts) + [len(order)]
    specs = [(int(cid), order[starts[i]:starts[i + 1]])
             for i, cid in enumerate(uniq)]

    def _write_one(spec):
        cid, idx_c = spec
        fname = f"part_{cid:04d}.parquet"
        table = pa.table({k: v[idx_c] for k, v in columns.items()})
        pq.write_table(table, prod_dir / fname, compression=_COMPRESSION)
        cx, cy, cz = chunk_coords(cid, n_side)
        return {
            "chunk_id": cid, "cx": cx, "cy": cy, "cz": cz,
            "xlo": float(lo[cid, 0]), "xhi": float(hi[cid, 0]),
            "ylo": float(lo[cid, 1]), "yhi": float(hi[cid, 1]),
            "zlo": float(lo[cid, 2]), "zhi": float(hi[cid, 2]),
            "n_rows": int(counts[cid]), "file": fname,
        }

    rows = map_partitions(_write_one, specs, n_threads=n_threads,
                          use_mpi=use_mpi)
    rows.sort(key=lambda r: r["chunk_id"])      # deterministic index order
    _write_index(prod_dir / INDEX_FILE, rows)
    return {"partition": "cartesian_chunk", "n_side": int(n_side),
            "n_chunks": int(len(uniq)), "n_rows": int(len(pos))}


# --------------------------------------------------------------------------
# Field product (regular grid) — memmap-able .npy tiles
# --------------------------------------------------------------------------
def _write_field_tiles(
    prod_dir: Path, field: np.ndarray, box_size: float, tile_cells: int,
) -> dict:
    prod_dir.mkdir(parents=True, exist_ok=True)
    n = field.shape[0]
    cell = box_size / n
    rows = []
    for s in tile_specs(n, tile_cells):
        tile = field[s["ix0"]:s["ix1"], s["iy0"]:s["iy1"], s["iz0"]:s["iz1"]]
        fname = f"tile_{s['tile_id']:04d}.npy"
        np.save(prod_dir / fname, np.ascontiguousarray(tile))
        rows.append({
            **s,
            "xlo": s["ix0"] * cell, "xhi": s["ix1"] * cell,
            "ylo": s["iy0"] * cell, "yhi": s["iy1"] * cell,
            "zlo": s["iz0"] * cell, "zhi": s["iz1"] * cell,
            "file": fname,
        })
    _write_index(prod_dir / INDEX_FILE, rows)
    return {"partition": "grid_tile", "n_grid": int(n),
            "tile_cells": int(tile_cells), "n_tiles": int(len(rows))}


# --------------------------------------------------------------------------
# Lightcone product — HEALPix NEST super-pixel partitions (like OUF)
# --------------------------------------------------------------------------
def _write_lightcone(prod_dir: Path, lc: Dict[str, np.ndarray], nside_part: int) -> dict:
    prod_dir.mkdir(parents=True, exist_ok=True)
    n = len(lc["lon"])
    if n == 0:
        _write_index(prod_dir / INDEX_FILE, [])
        return {"partition": "healpix_nest", "nside_part": int(nside_part),
                "n_pixels": 0, "n_rows": 0}
    superpix = healpix_partition_ids(lc["lon"], lc["lat"], nside_part)
    order = np.argsort(superpix, kind="stable")
    sp_sorted = superpix[order]
    cols_sorted = {k: np.asarray(v)[order] for k, v in lc.items()}

    uniq, starts = np.unique(sp_sorted, return_index=True)
    starts = list(starts) + [len(sp_sorted)]
    rows = []
    for i, pix in enumerate(uniq):
        sl = slice(starts[i], starts[i + 1])
        fname = f"part_{int(pix):04d}.parquet"
        table = pa.table({k: v[sl] for k, v in cols_sorted.items()})
        pq.write_table(table, prod_dir / fname, compression=_COMPRESSION)
        rows.append({"pixel": int(pix), "nside_part": int(nside_part),
                     "n_rows": int(sl.stop - sl.start), "file": fname})
    _write_index(prod_dir / INDEX_FILE, rows)
    return {"partition": "healpix_nest", "nside_part": int(nside_part),
            "n_pixels": int(len(uniq)), "n_rows": int(n)}


# --------------------------------------------------------------------------
# Native readers (linear-sim layout)
# --------------------------------------------------------------------------
def _read_parquet_cols(path: Path) -> Dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy(zero_copy_only=False)
            for name in table.column_names}


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def write_oufsim_store(
    native_dir: Union[str, Path],
    out_root: Union[str, Path],
    *,
    sim_name: str,
    sim_kind: str = "pm",
    particle_chunk_nside: int = 4,
    field_tile_cells: int = 32,
    lightcone_nside_part: int = 2,
    batch_rows: Optional[int] = None,
    n_threads: int = 1,
    use_mpi: bool = False,
    plan: Optional[ExecutionPlan] = None,
    overwrite: bool = False,
) -> Path:
    """Convert a native linear-sim ``native_dir`` to an OUF-Sim store.

    Returns the ``…/{sim_name}/oufsim`` directory. If an ``ExecutionPlan`` is
    given, its mode is enforced against the writer's declared capabilities
    (unsupported mode → ValueError, never a silent fallback — Rule 5).
    """
    if plan is not None:
        for step in _HEAVY_STEPS:
            if not _WRITE_CAPS.supports(step, plan.mode):
                raise ValueError(
                    f"{step}: ExecutionMode.{plan.mode.name} not supported "
                    f"by the OUF-Sim writer (allowed: "
                    f"{[m.name for m in _WRITE_CAPS.modes_for(step)]})"
                )
        if plan.mode == ExecutionMode.MPI:
            use_mpi = True
        if plan.batch_rows is not None:
            batch_rows = plan.batch_rows
    native_dir = Path(native_dir)
    cfg = yaml.safe_load((native_dir / "config.yaml").read_text())
    box_size = float(cfg["box_size"])
    n_grid = int(cfg["n_grid"])
    redshifts = [float(z) for z in cfg["redshifts"]]
    cosmo = CosmologySpec.from_dict(cfg["cosmology"])

    store = Path(out_root) / sim_name / OUFSIM_SUBDIR
    if store.exists():
        if not overwrite:
            raise FileExistsError(f"{store} exists; pass overwrite=True")
        import shutil
        shutil.rmtree(store)
    store.mkdir(parents=True)

    layout: Dict[str, dict] = {"snapshots": {}, "fields": {}, "halos": {}}
    n_particles_total = 0

    for z in redshifts:
        zt = _ztag(z)
        zdir = native_dir / zt

        parts = np.load(zdir / "particles.npy")  # (Np, 6)
        pcols = {"x": parts[:, 0], "y": parts[:, 1], "z": parts[:, 2],
                 "vx": parts[:, 3], "vy": parts[:, 4], "vz": parts[:, 5]}
        layout["snapshots"][zt] = _write_chunked_catalog(
            store / "snapshots" / zt, pcols, parts[:, :3],
            box_size, particle_chunk_nside, batch_rows=batch_rows,
            n_threads=n_threads, use_mpi=use_mpi,
        )
        layout["snapshots"][zt]["dir"] = f"snapshots/{zt}"
        layout["snapshots"][zt]["index"] = f"snapshots/{zt}/{INDEX_FILE}"
        n_particles_total = max(n_particles_total, parts.shape[0])

        field = np.load(zdir / "field.npy")
        layout["fields"][zt] = _write_field_tiles(
            store / "fields" / zt, field, box_size, field_tile_cells,
        )
        layout["fields"][zt]["dir"] = f"fields/{zt}"
        layout["fields"][zt]["index"] = f"fields/{zt}/{INDEX_FILE}"

        halos = _read_parquet_cols(zdir / "halos.parquet")
        if len(halos["x"]) > 0:
            hpos = np.stack([halos["x"], halos["y"], halos["z"]], axis=1)
            layout["halos"][zt] = _write_chunked_catalog(
                store / "halos" / zt, halos, hpos,
                box_size, max(1, particle_chunk_nside // 2),
                batch_rows=batch_rows, n_threads=n_threads, use_mpi=use_mpi,
            )
            layout["halos"][zt]["dir"] = f"halos/{zt}"
            layout["halos"][zt]["index"] = f"halos/{zt}/{INDEX_FILE}"

    products = ["snapshots", "fields", "halos"]

    tree_path = native_dir / "tree.parquet"
    if tree_path.is_file():
        tree = _read_parquet_cols(tree_path)
        tdir = store / "tree"
        tdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(tree), tdir / "part_0000.parquet",
                       compression=_COMPRESSION)
        n_edges = int(len(next(iter(tree.values())))) if tree else 0
        _write_index(tdir / INDEX_FILE,
                     [{"partition": 0, "n_rows": n_edges,
                       "file": "part_0000.parquet"}])
        layout["tree"] = {"partition": "single", "n_rows": n_edges,
                          "dir": "tree", "index": f"tree/{INDEX_FILE}"}
        products.append("tree")

    lc_path = native_dir / "lightcone.parquet"
    if lc_path.is_file():
        lc = _read_parquet_cols(lc_path)
        lc_info = _write_lightcone(store / "lightcone", lc, lightcone_nside_part)
        lc_info["dir"] = "lightcone"
        lc_info["index"] = f"lightcone/{INDEX_FILE}"
        layout["lightcone"] = lc_info
        products.append("lightcone")

    decls = (
        ProductDecl("snapshots", "linear .npy (x,y,z,vx,vy,vz)",
                    ("cartesian_chunk",), ("x", "y", "z", "vx", "vy", "vz")),
        ProductDecl("fields", "linear .npy mesh", ("grid_tile",), ("delta",)),
        ProductDecl("halos", "linear parquet", ("cartesian_chunk",),
                    ("halo_id", "x", "y", "z", "delta_peak", "mass")),
    )
    if "tree" in products:
        decls = decls + (ProductDecl(
            "tree", "linear parquet (edges)", ("single",),
            ("descendant_id", "progenitor_id", "z_desc", "z_prog"),
        ),)
    if "lightcone" in products:
        decls = decls + (ProductDecl(
            "lightcone", "linear parquet (sky)", ("healpix_nest",),
            ("lon", "lat", "redshift", "comoving_radius", "mass", "_healpix32"),
        ),)

    manifest = OUFSimManifest(
        oufsim_format_version=OUFSIM_FORMAT_VERSION,
        sim_name=sim_name,
        sim_kind=sim_kind,
        code="oneuniverse.simulation.linear",
        code_version=cfg.get("generator"),
        layout_schema="per_cosmology_phase_snapshot",
        backends=("linear",),
        has_input=False,
        has_output=True,
        products=tuple(products),
        n_snapshots=len(redshifts),
        redshifts=tuple(redshifts),
        box_size=box_size,
        n_particles=int(n_particles_total),
        cosmology=cosmo,
        unit_frame=UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar", comoving=True, frame="box",
        ),
        provenance=ProvenanceSpec(
            code="oneuniverse.simulation.linear",
            code_version=cfg.get("generator"),
            git_hash=None,
            original_paths=(str(native_dir),),
            ingested_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            converter="LinearSimConverter",
        ),
    )
    payload = manifest.to_dict()
    payload["store_layout"] = layout
    payload["n_grid"] = n_grid
    write_json(store / "manifest.json", payload)
    return store
