"""Random "native simulation" directories shared by the simulation-store tests.

No physics: uniform positions, Gaussian velocities, white-noise fields.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.simulation.manifest import OUFSIM_FORMAT_VERSION, OUFSimManifest
from oneuniverse.simulation.specs import (
    CosmologySpec,
    ProductDecl,
    ProvenanceSpec,
    UnitFrameSpec,
)
from oneuniverse.simulation.store.build import NativeProduct
from oneuniverse.simulation.store.index import chunk_ids

PARTICLE_COLUMNS = ("x", "y", "z", "vx", "vy", "vz")
_COSMOLOGY = {"h": 0.67, "omega_m": 0.31, "omega_b": 0.049, "sigma8": 0.81, "n_s": 0.965}
_BATCH = 20_000


def _tag(z: float) -> str:
    return f"{z:.3f}"


def make_native_sim(out, *, n_particles=50_000, n_grid=32, box_size=100.0,
                    redshifts=(1.0, 0.0), seed=0) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_halos = max(50, n_particles // 100)
    for z in redshifts:
        tag = _tag(z)
        arr = np.empty(n_particles, dtype=[(c, np.float32) for c in PARTICLE_COLUMNS])
        for c in "xyz":
            arr[c] = rng.uniform(0, box_size, n_particles).astype(np.float32)
        for c in ("vx", "vy", "vz"):
            arr[c] = rng.normal(0, 300, n_particles).astype(np.float32)
        np.save(out / f"particles_z{tag}.npy", arr)
        np.save(out / f"field_z{tag}.npy",
                rng.normal(0, 1, (n_grid,) * 3).astype(np.float32))
        halos = {c: rng.uniform(0, box_size, n_halos) for c in "xyz"}
        halos["mass"] = 10 ** rng.uniform(11, 15, n_halos)
        for c in ("vx", "vy", "vz"):
            halos[c] = rng.normal(0, 300, n_halos)
        halos["radius"] = rng.uniform(0.1, 2.0, n_halos)
        pq.write_table(pa.table(halos), out / f"halos_z{tag}.parquet")
    n_lc = max(100, n_particles // 5)
    lc = {"ra": rng.uniform(0, 360, n_lc), "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n_lc))),
          "z": rng.uniform(0, 1.5, n_lc), "mass": 10 ** rng.uniform(11, 15, n_lc)}
    pq.write_table(pa.table(lc), out / "lightcone.parquet")
    (out / "config.json").write_text(json.dumps({
        "box_size": box_size, "n_grid": n_grid, "redshifts": list(redshifts), "seed": seed,
        "cosmology": dict(_COSMOLOGY)}))
    return out


def _config(path: Path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def native_manifest(path) -> OUFSimManifest:
    """Manifest without products (``build_store`` fills them)."""
    path = Path(path)
    cfg = _config(path)
    return OUFSimManifest(
        OUFSIM_FORMAT_VERSION, "native_sim", cfg["box_size"], CosmologySpec(**cfg["cosmology"]),
        UnitFrameSpec(), ProvenanceSpec("fixture", "1", cfg["seed"], "fixture"), (),
        {"format": "npy", "root": str(path), "projection": "reencode"}, "2026-11-12T00:00:00Z")


def _npy_batches(file: Path):
    arr = np.load(file, mmap_mode="r")
    for s in range(0, len(arr), _BATCH):
        chunk = arr[s:s + _BATCH]
        yield pa.RecordBatch.from_arrays(
            [pa.array(np.ascontiguousarray(chunk[c])) for c in PARTICLE_COLUMNS],
            names=list(PARTICLE_COLUMNS))


def _parquet_batches(file: Path):
    yield from pq.ParquetFile(file).iter_batches(batch_size=_BATCH)


def native_products(path, *, adapter_format: str | None = None) -> list[NativeProduct]:
    path = Path(path)
    cfg = _config(path)
    zs = tuple(float(z) for z in cfg["redshifts"])
    tags = {z: _tag(z) for z in zs}
    halo_cols = tuple(pq.read_schema(path / f"halos_z{tags[zs[0]]}.parquet").names)
    lc_file = path / "lightcone.parquet"
    lc_cols = tuple(pq.read_schema(lc_file).names)
    part_src = {z: path / f"particles_z{t}.npy" for z, t in tags.items()}
    halo_src = {z: path / f"halos_z{t}.parquet" for z, t in tags.items()}
    return [
        NativeProduct(ProductDecl("particles", "particles", zs, PARTICLE_COLUMNS),
                      "catalog", part_src, adapter_format,
                      lambda z, s=part_src: _npy_batches(s[float(z)])),
        NativeProduct(ProductDecl("halos", "halos", zs, halo_cols), "catalog", halo_src, None,
                      lambda z, s=halo_src: _parquet_batches(s[float(z)])),
        NativeProduct(ProductDecl("density", "field", zs, n_grid=cfg["n_grid"]), "field",
                      {z: path / f"field_z{t}.npy" for z, t in tags.items()}),
        NativeProduct(ProductDecl("lightcone", "lightcone", (0.0,), lc_cols), "lightcone",
                      {0.0: lc_file}, None, lambda z: _parquet_batches(lc_file)),
    ]


def write_packed_native_fixture(src, *, n_side: int) -> Path:
    """Copy ``src`` to ``<src>_packed`` with particles sorted by chunk id plus ``header.json``."""
    src = Path(src)
    dst = src.with_name(src.name + "_packed")
    dst.mkdir(parents=True, exist_ok=True)
    cfg = _config(src)
    chunks = {}
    for z in cfg["redshifts"]:
        tag = _tag(z)
        arr = np.load(src / f"particles_z{tag}.npy")
        pos = np.stack([arr[c] for c in "xyz"], axis=1).astype(np.float64)
        ids = chunk_ids(pos, cfg["box_size"], n_side)
        order = np.argsort(ids, kind="stable")
        np.save(dst / f"particles_z{tag}.npy", arr[order])
        uniq, start, count = np.unique(ids[order], return_index=True, return_counts=True)
        chunks[tag] = [[int(c), int(a), int(a + n)] for c, a, n in zip(uniq, start, count)]
        shutil.copy2(src / f"field_z{tag}.npy", dst / f"field_z{tag}.npy")
        shutil.copy2(src / f"halos_z{tag}.parquet", dst / f"halos_z{tag}.parquet")
    for name in ("lightcone.parquet", "config.json"):
        shutil.copy2(src / name, dst / name)
    (dst / "header.json").write_text(json.dumps({
        "box_size": cfg["box_size"], "n_side": n_side, "redshifts": cfg["redshifts"],
        "chunks": chunks}))
    return dst
