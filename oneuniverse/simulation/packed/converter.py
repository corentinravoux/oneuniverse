"""PackedSimConverter — 2nd backend over the packed_npy native format.

Demonstrates that the store machinery is not coupled to the linear layout: a
new code is added by (1) a NativeReaderAdapter (here PackedNpyAdapter) and
(2) a SimConverter that emits NativeProduct descriptors + calls build_store.
A real backend (AbacusSummit ASDF, Gadget HDF5) follows the same recipe.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.converter import SimConverter, register
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode
from oneuniverse.simulation.oufsim.build import NativeProduct, build_store
from oneuniverse.simulation.oufsim.native import get_adapter
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def _load_packed_catalog(slab_path: Path) -> dict:
    arr = np.load(slab_path)          # (N,6) chunk-sorted
    cols = ("x", "y", "z", "vx", "vy", "vz")
    return {c: arr[:, j] for j, c in enumerate(cols)}


def _load_packed_field(field_path: Path) -> np.ndarray:
    return np.load(field_path)


def _load_packed_halos(halo_path: Path) -> dict:
    t = pq.read_table(halo_path)
    return {n: t.column(n).to_numpy(zero_copy_only=False)
            for n in t.column_names}


def _load_lightcone(lc_path: Path) -> dict:
    t = pq.read_table(lc_path)
    return {n: t.column(n).to_numpy(zero_copy_only=False)
            for n in t.column_names}


@register
class PackedSimConverter(SimConverter):
    code = "packed_npy"
    sim_kind = "pm"
    capabilities = BackendCapabilities(
        name="packed_npy",
        native_format="packed_npy (chunk-sorted slab + header.json)",
        supports_random_access=True, supports_streaming=True,
        heavy_step_modes={
            "particle_chunking": (ExecutionMode.SEQUENTIAL,),
            "field_tiling": (ExecutionMode.SEQUENTIAL,),
        },
    )

    def detect(self, path: Path) -> bool:
        hdr = Path(path) / "header.json"
        if not hdr.is_file():
            return False
        try:
            raw = json.loads(hdr.read_text())
        except json.JSONDecodeError:
            return False
        return raw.get("native_format") == "packed_npy"

    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        return (
            ProductDecl("snapshots", "packed_npy slab", ("cartesian_chunk",),
                        ("x", "y", "z", "vx", "vy", "vz")),
            ProductDecl("fields", "packed_npy mesh", ("grid_tile",), ("delta",)),
            ProductDecl("halos", "packed parquet", ("cartesian_chunk",),
                        ("halo_id", "x", "y", "z", "delta_peak", "mass")),
        )

    def read_cosmology(self, src: Path) -> CosmologySpec:
        raw = json.loads((Path(src) / "header.json").read_text())
        return CosmologySpec.from_dict(raw["cosmology"])

    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        return UnitFrameSpec(length_unit="Mpc/h", mass_unit="Msun/h",
                             velocity_unit="km/s peculiar", comoving=True,
                             frame="box")

    def convert(self, src: Path, out: Path, *, projection: str = "reencode",
                build_indexes: bool = True, sim_name: str = "packsim",
                overwrite: bool = False, **kwargs) -> Path:
        src = Path(src)
        hdr = json.loads((src / "header.json").read_text())
        get_adapter("packed_npy")          # validate the format is registered
        box = float(hdr["box_size"]); n_grid = int(hdr["n_grid"])
        redshifts = [float(z) for z in hdr["redshifts"]]
        products = []
        for zt, blk in hdr["snapshots"].items():
            z = float(zt[1:])
            if projection == "reference":
                products.append(NativeProduct(
                    name="snapshots", kind="catalog", z=z, load=lambda: None,
                    columns=("x", "y", "z", "vx", "vy", "vz"),
                    n_side=int(blk.get("n_side", 4)), projection="reference",
                    native_path=str(src / blk["file"]),
                    chunk_index=blk["chunk_index"]))
            else:
                products.append(NativeProduct(
                    name="snapshots", kind="catalog", z=z,
                    load=partial(_load_packed_catalog, src / blk["file"]),
                    columns=("x", "y", "z", "vx", "vy", "vz"),
                    n_side=int(blk.get("n_side", 4))))
        for zt, blk in hdr["fields"].items():
            products.append(NativeProduct(
                name="fields", kind="field", z=float(zt[1:]),
                load=partial(_load_packed_field, src / blk["file"])))
        for zt, blk in hdr.get("halos", {}).items():
            products.append(NativeProduct(
                name="halos", kind="catalog", z=float(zt[1:]),
                load=partial(_load_packed_halos, src / blk["file"]),
                columns=("halo_id", "x", "y", "z", "delta_peak", "mass"),
                n_side=2))
        if hdr.get("lightcone"):
            products.append(NativeProduct(
                name="lightcone", kind="lightcone", z=None,
                load=partial(_load_lightcone, src / hdr["lightcone"]["file"]),
                nside_part=2))
        return build_store(
            out, sim_name=sim_name, cosmo=self.read_cosmology(src),
            unit_frame=self.read_unit_frame(src), box_size=box, n_grid=n_grid,
            redshifts=redshifts, products=products, code=self.code,
            sim_kind=self.sim_kind, native_format="packed_npy",
            overwrite=overwrite)
