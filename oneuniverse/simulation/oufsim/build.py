"""Format-agnostic OUF-Sim store builder.

A converter describes its products as ``NativeProduct`` descriptors (a name,
a kind, a redshift, and a ``load`` callable that returns the native arrays via
*any* mechanism — direct numpy, a NativeReaderAdapter, an HDF5 reader). The
builder loops them and calls the **same** per-product writers as the linear
``write_oufsim_store``, then emits the manifest. This is the seam a real
backend reuses: implement an adapter + emit ``NativeProduct``s, get a store.

Core kinds: ``catalog`` (particles/halos, Cube-chunked), ``field`` (memmap
tiles or reference), ``lightcone`` (HEALPix-NEST). Extended products stay in
the linear writer (out of S17 scope).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from oneuniverse.simulation._version import OUFSIM_FORMAT_VERSION
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.manifest import OUFSimManifest
from oneuniverse.simulation.oufsim._io import write_json
from oneuniverse.simulation.oufsim._layout import write_store_layout
from oneuniverse.simulation.oufsim.write import (
    INDEX_FILE, OUFSIM_SUBDIR, _write_chunked_catalog,
    _write_chunked_catalog_reference, _write_field_reference,
    _write_field_tiles, _write_lightcone,
)
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec


@dataclass
class NativeProduct:
    name: str
    kind: str                                  # "catalog" | "field" | "lightcone"
    z: Optional[float]
    load: Callable[[], object]                 # -> dict cols | ndarray
    columns: Tuple[str, ...] = ()
    pos_keys: Tuple[str, str, str] = ("x", "y", "z")
    n_side: int = 4                            # catalog chunking
    tile_cells: int = 32                       # field tiling
    nside_part: int = 2                        # lightcone partition
    projection: str = "reencode"               # "reencode" | "reference"
    native_path: Optional[str] = None          # field/catalog reference target
    chunk_index: Optional[List[dict]] = None   # catalog reference (sorted slab)


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def build_store(
    out_root: Union[str, Path], *, sim_name: str, cosmo: CosmologySpec,
    unit_frame: UnitFrameSpec, box_size: float, n_grid: int,
    redshifts: Sequence[float], products: Sequence[NativeProduct],
    code: str, sim_kind: str = "pm", native_format: Optional[str] = None,
    n_threads: int = 1, overwrite: bool = False,
) -> Path:
    """Build an OUF-Sim store from product descriptors. Returns the store dir."""
    store = Path(out_root) / sim_name / OUFSIM_SUBDIR
    if store.exists():
        if not overwrite:
            raise FileExistsError(f"{store} exists; pass overwrite=True")
        import shutil
        shutil.rmtree(store)
    store.mkdir(parents=True)

    layout: Dict[str, dict] = {}
    product_names: List[str] = []
    n_particles_total = 0

    for p in products:
        zt = _ztag(float(p.z)) if p.z is not None else None
        if p.kind == "catalog":
            if p.projection == "reference":
                info = _write_chunked_catalog_reference(
                    store / p.name / zt, p.chunk_index, p.native_path,
                    p.columns)
            else:
                cols = p.load()
                pos = np.stack([cols[k] for k in p.pos_keys], axis=1)
                info = _write_chunked_catalog(
                    store / p.name / zt, cols, pos, box_size, p.n_side,
                    n_threads=n_threads)
            info["dir"] = f"{p.name}/{zt}"
            info["index"] = f"{p.name}/{zt}/{INDEX_FILE}"
            layout.setdefault(p.name, {})[zt] = info
            if p.name == "snapshots":
                n_particles_total = max(n_particles_total, info["n_rows"])
        elif p.kind == "field":
            if p.projection == "reference":
                info = _write_field_reference(
                    store / p.name / zt, Path(p.native_path), n_grid, box_size)
            else:
                info = _write_field_tiles(
                    store / p.name / zt, np.asarray(p.load()), box_size,
                    p.tile_cells)
            info["dir"] = f"{p.name}/{zt}"
            info["index"] = f"{p.name}/{zt}/{INDEX_FILE}"
            layout.setdefault(p.name, {})[zt] = info
        elif p.kind == "lightcone":
            info = _write_lightcone(store / p.name, p.load(), p.nside_part)
            info["dir"] = p.name
            info["index"] = f"{p.name}/{INDEX_FILE}"
            layout[p.name] = info
        else:
            raise ValueError(f"build_store: unknown kind {p.kind!r}")
        if p.name not in product_names:
            product_names.append(p.name)

    manifest = OUFSimManifest(
        oufsim_format_version=OUFSIM_FORMAT_VERSION, sim_name=sim_name,
        sim_kind=sim_kind, code=code, code_version=None,
        layout_schema="per_cosmology_phase_snapshot", backends=(code,),
        has_input=False, has_output=True, products=tuple(product_names),
        n_snapshots=len(tuple(redshifts)),
        redshifts=tuple(float(z) for z in redshifts),
        box_size=float(box_size), n_particles=int(n_particles_total) or None,
        cosmology=cosmo, unit_frame=unit_frame,
        provenance=ProvenanceSpec(
            code=code, code_version=None, git_hash=None, original_paths=(),
            ingested_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            converter="build_store"),
    )
    payload = manifest.to_dict()
    payload["n_grid"] = int(n_grid)
    if native_format is not None:
        payload["native_format"] = native_format
    write_json(store / "manifest.json", payload)
    write_store_layout(store, layout)  # S11: layout lives in its own sidecar
    return store
