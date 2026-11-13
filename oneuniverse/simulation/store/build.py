"""Store construction: :class:`NativeProduct` description and :func:`build_store`."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.fs import atomic_dir, atomic_write_json
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    OUFSimManifest,
    write_manifest,
)
from oneuniverse.simulation.specs import ProductDecl
from oneuniverse.simulation.store.native import NativeReaderAdapter, get_adapter
from oneuniverse.simulation.store.read import SimStore
from oneuniverse.simulation.store.write import (
    schema_row_bytes,
    write_catalog_product,
    write_field_product,
    write_lightcone_product,
    write_reference_catalog,
    write_reference_field,
)


@dataclass
class NativeProduct:
    """One product of a native simulation, ready to be written into a store.

    ``sources`` maps redshift to the native file; ``adapter_format`` names the
    native reader adapter (``None``: default for the layout); ``batches`` yields
    record batches of a catalog or lightcone product at a redshift.
    """

    decl: ProductDecl
    kind_layout: Literal["catalog", "field", "lightcone"]
    sources: dict[float, Path]
    adapter_format: str | None = None
    batches: Callable[[float], Iterable[pa.RecordBatch]] | None = None


def _native_batches(adapter: NativeReaderAdapter, path: Path, plan: ExecutionPlan) -> Iterator[pa.RecordBatch]:
    n = adapter.num_rows(path)
    probe = adapter.read_rows(path, 0, min(n, 1))
    step = plan.batch_for(schema_row_bytes(probe.schema))
    for s in range(0, n, step):
        yield from adapter.read_rows(path, s, min(n, s + step)).to_batches()


def build_store(out, *, manifest: OUFSimManifest, products: Sequence[NativeProduct],
                plan: ExecutionPlan, projection: Literal["reencode", "reference"] = "reencode",
                n_side: int = 8, tile_cells: int = 32, nside_part: int = 1) -> SimStore:
    """Build a store at ``out`` (published atomically) from native products.

    ``reencode`` rewrites every product into store chunks/tiles/pixels.
    ``reference`` writes index-only products that point at the native files; it
    applies to catalog products whose adapter is ``packed_npy`` (chunk-sorted
    native slab) and to field products (``.npy`` grids).  Everything
    else (halos in parquet, lightcones, unsorted particle files) is re-encoded
    regardless.  Catalog rows come from ``NativeProduct.batches`` when given,
    otherwise from the native adapter in ``plan.batch_for``-sized slices.
    ``manifest.products`` is ignored and rebuilt from the declarations (field
    ``n_grid`` taken from the native grid).
    """
    if projection not in ("reencode", "reference"):
        raise ValidationError(f"projection must be 'reencode' or 'reference', got {projection!r}")
    out = Path(out)
    decls: list[ProductDecl] = []
    with atomic_dir(out) as tmp:
        atomic_write_json(tmp / LAYOUT_FILENAME, {"products": {}})
        for nat in products:
            decl = nat.decl
            for z in decl.redshifts:
                src = nat.sources.get(float(z))
                ref = projection == "reference"
                if nat.kind_layout == "catalog":
                    if ref and nat.adapter_format == "packed_npy" and src is not None:
                        write_reference_catalog(tmp, decl.name, z, src, get_adapter("packed_npy"),
                                                box_size=manifest.box_size, n_side=n_side)
                        continue
                    if nat.batches is not None:
                        batches = nat.batches(z)
                    elif src is not None:
                        batches = _native_batches(get_adapter(nat.adapter_format or "npy"), src, plan)
                    else:
                        raise ValidationError(f"product {decl.name!r} z={z}: no batches and no source file")
                    write_catalog_product(tmp, decl.name, z, batches, box_size=manifest.box_size,
                                          n_side=n_side, plan=plan)
                elif nat.kind_layout == "field":
                    if src is None:
                        raise ValidationError(f"field {decl.name!r} z={z}: no source file")
                    adapter = get_adapter(nat.adapter_format or "npy")
                    shape = adapter.field_shape(src)
                    if decl.n_grid is not None and decl.n_grid != shape[0]:
                        raise ValidationError(
                            f"field {decl.name!r}: declared n_grid={decl.n_grid}, native grid is {shape[0]}")
                    decl = replace(decl, n_grid=shape[0])
                    if ref:
                        write_reference_field(tmp, decl.name, z, src, adapter, tile_cells=tile_cells)
                    else:
                        write_field_product(tmp, decl.name, z, np.load(src, mmap_mode="r"),
                                            tile_cells=tile_cells, plan=plan)
                elif nat.kind_layout == "lightcone":
                    if nat.batches is None:
                        raise ValidationError(f"lightcone {decl.name!r} needs NativeProduct.batches")
                    write_lightcone_product(tmp, decl.name, nat.batches(z), nside_part=nside_part,
                                            plan=plan, z=z)
                else:
                    raise ValidationError(f"unknown kind_layout {nat.kind_layout!r}")
            decls.append(decl)
        native = dict(manifest.native)
        native["projection"] = projection
        write_manifest(tmp / MANIFEST_FILENAME,
                       replace(manifest, products=tuple(decls), native=native))
    return SimStore(out)
