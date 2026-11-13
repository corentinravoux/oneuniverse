"""Streaming writers of the partial-access simulation store.

Layout under ``<root>``::

    manifest.json, layout.json
    products/<name>/z=<tag>/_index.parquet
                           /chunk_<id>.parquet   (catalog, one file per non-empty chunk)
                           /tile_<id>.npy        (field)
                           /pix_<id>.parquet     (lightcone, NEST ``nside_part``)

``<tag>`` is ``f"{z:.3f}"``.  Every writer builds ``z=<tag>.tmp`` and renames
it into place, then merges its entry into ``layout.json`` (atomically); an
existing product at that redshift is a ``FormatError``.

Row numbering: catalog ``row_start``/``row_stop`` are cumulative over chunks in
chunk-id order (a global numbering); empty chunks have an index row with
``row_start == row_stop``, an empty ``path`` and NaN bounding box.  In a
re-encoded chunk file rows are local, ``0 .. row_stop - row_start``; in a
reference store they address the native (chunk-sorted) file directly.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.fs import atomic_write_json
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    read_manifest,
    write_manifest,
)
from oneuniverse.simulation.store.index import (
    CATALOG_INDEX_SCHEMA,
    FIELD_INDEX_SCHEMA,
    SKY_INDEX_SCHEMA,
    chunk_bbox,
    chunk_ids,
)
from oneuniverse.simulation.store.native import NativeReaderAdapter

INDEX_FILENAME = "_index.parquet"
_MAX_OPEN = 256  # most ParquetWriters open at once (1024 is a common fd limit)
# Chunk files are located through ``_index.parquet``, never by predicate pushdown,
# so row-group statistics and the embedded arrow schema are dropped: per-file
# metadata is what dominates small chunks.
_PQ_OPTS = {"write_statistics": False, "store_schema": False, "use_dictionary": False,
            "compression": "zstd"}


def ztag(z: float) -> str:
    return f"{float(z):.3f}"


# --------------------------------------------------------------------- layout

def _read_layout(store_root: Path) -> dict:
    p = Path(store_root) / LAYOUT_FILENAME
    if not p.exists():
        return {"products": {}}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise FormatError(f"{p} is not valid JSON: {exc}") from exc


def _merge_layout(store_root: Path, name: str, tag: str, entry: dict) -> None:
    layout = _read_layout(store_root)
    layout.setdefault("products", {}).setdefault(name, {})[tag] = entry
    atomic_write_json(Path(store_root) / LAYOUT_FILENAME, layout)


def _product_dir(store_root: Path, name: str, tag: str) -> Path:
    return Path(store_root) / "products" / name / f"z={tag}"


def _check_new(store_root: Path, name: str, tag: str) -> None:
    if tag in _read_layout(store_root).get("products", {}).get(name, {}) \
            or _product_dir(store_root, name, tag).exists():
        raise FormatError(f"product {name!r} at z={tag} already exists in {store_root}")


@contextlib.contextmanager
def _staged_product(store_root: Path, name: str, tag: str) -> Iterator[Path]:
    """Yield an empty ``z=<tag>.tmp`` dir; rename it into place on success."""
    _check_new(store_root, name, tag)
    final = _product_dir(store_root, name, tag)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        raise FormatError(f"stale staging directory {tmp}: delete it if no writer is running")
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    os.rename(tmp, final)


def _rel_dir(name: str, tag: str) -> str:
    return f"products/{name}/z={tag}"


# ----------------------------------------------------------- partitioned write

def _as_table(batch: pa.RecordBatch | pa.Table, schema: pa.Schema | None) -> pa.Table:
    table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
    if schema is not None and table.schema != schema:
        try:
            table = table.select(schema.names).cast(schema)
        except (KeyError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise ValidationError(
                f"batch schema {table.schema.names} does not match {schema.names}: {exc}") from exc
    return table


def _groups(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable sort order, unique keys, group start offsets (into the sorted order)."""
    order = np.argsort(keys, kind="stable")
    uniq, start = np.unique(keys[order], return_index=True)
    return order, uniq, start


def _partition_write(batches: Iterable, key_fn: Callable[[pa.RecordBatch], np.ndarray],
                     lo: int, hi: int, path_for: Callable[[int], Path], stage_root: Path,
                     batch_rows: Callable[[pa.Schema], int], *,
                     observe: Callable[[pa.RecordBatch, np.ndarray], None] | None = None,
                     depth: int = 0) -> None:
    """Write rows with key in ``[lo, hi)`` to ``path_for(key)``, <= ``_MAX_OPEN`` files open.

    Direct when the key range is small enough; otherwise rows are first staged
    in at most ``_MAX_OPEN`` key-range buckets (parquet files in ``stage_root``)
    and each bucket is partitioned recursively.
    """
    n = hi - lo
    schema: pa.Schema | None = None
    with contextlib.ExitStack() as stack:
        writers: dict[int, pq.ParquetWriter] = {}
        width = 1 if n <= _MAX_OPEN else math.ceil(n / _MAX_OPEN)
        for batch in batches:
            if batch.num_rows == 0:
                continue
            keys = np.asarray(key_fn(batch))
            if observe is not None:
                observe(batch, keys)
            table = _as_table(batch, schema)
            if schema is None:
                schema = table.schema
            wkeys = keys if width == 1 else (keys - lo) // width + lo
            order, uniq, start = _groups(wkeys)
            sorted_table = table.take(pa.array(order))
            stops = np.append(start[1:], len(order))
            for k, a, b in zip(uniq.tolist(), start.tolist(), stops.tolist()):
                if k not in writers:
                    path = path_for(k) if width == 1 else stage_root / f"s{depth}_{k - lo}.parquet"
                    writers[k] = stack.enter_context(pq.ParquetWriter(path, schema, **_PQ_OPTS))
                writers[k].write_table(sorted_table.slice(a, b - a))
    if width == 1:
        return
    for b in range(math.ceil(n / width)):
        sub_lo, sub_hi = lo + b * width, min(hi, lo + (b + 1) * width)
        stage = stage_root / f"s{depth}_{b}.parquet"
        if not stage.exists():
            continue
        pf = pq.ParquetFile(stage)
        rows = batch_rows(pf.schema_arrow)
        _partition_write(pf.iter_batches(batch_size=rows), key_fn, sub_lo, sub_hi, path_for,
                         stage_root, batch_rows, depth=depth + 1)
        del pf
        stage.unlink()


def _row_bytes(schema: pa.Schema) -> int:
    total = 0
    for f in schema:
        try:
            total += max(1, f.type.bit_width // 8)
        except ValueError:
            total += 16  # variable width: rough estimate
    return max(total, 1)


# -------------------------------------------------------------------- catalog

def write_catalog_product(store_root, name: str, z: float, batches: Iterable[pa.RecordBatch], *,
                          box_size: float, n_side: int, plan: ExecutionPlan | None = None) -> dict:
    """Stream ``batches`` into per-chunk parquet files plus ``_index.parquet``.

    ``batches`` need float columns ``x``, ``y``, ``z`` (positions, wrapped into
    the box).  Memory is bounded by one batch plus the open writers; with more
    than ``_MAX_OPEN`` chunks a two-pass key-range bucketing keeps open files
    bounded.  Returns the layout entry.
    """
    plan = plan or ExecutionPlan()
    store_root, tag = Path(store_root), ztag(z)
    n_chunks = int(n_side) ** 3
    counts = np.zeros(n_chunks, dtype=np.int64)
    bmin = np.full((n_chunks, 3), np.inf)
    bmax = np.full((n_chunks, 3), -np.inf)

    def positions(batch: pa.RecordBatch) -> np.ndarray:
        try:
            return np.stack([batch.column(a).to_numpy(zero_copy_only=False).astype(np.float64)
                             for a in "xyz"], axis=1)
        except KeyError:
            raise ValidationError(
                f"catalog batches need position columns x, y, z; got {batch.schema.names}") from None

    def key_fn(batch: pa.RecordBatch) -> np.ndarray:
        return chunk_ids(positions(batch), box_size, n_side)

    def observe(batch: pa.RecordBatch, keys: np.ndarray) -> None:
        pos = positions(batch)
        order, uniq, start = _groups(keys)
        counts[uniq] += np.diff(np.append(start, len(order)))
        sp = pos[order]
        bmin[uniq] = np.minimum(bmin[uniq], np.minimum.reduceat(sp, start, axis=0))
        bmax[uniq] = np.maximum(bmax[uniq], np.maximum.reduceat(sp, start, axis=0))

    with _staged_product(store_root, name, tag) as tmp:
        stage = tmp / "_stage"
        stage.mkdir()
        _partition_write(
            batches, key_fn, 0, n_chunks, lambda k: tmp / f"chunk_{k}.parquet", stage,
            lambda s: max(1, plan.batch_for(_row_bytes(s))), observe=observe)
        shutil.rmtree(stage)
        stop = np.cumsum(counts)
        start = stop - counts
        paths = [f"chunk_{c}.parquet" if counts[c] else "" for c in range(n_chunks)]
        empty = counts == 0
        bmin[empty] = np.nan
        bmax[empty] = np.nan
        cols = [pa.array(np.arange(n_chunks, dtype=np.int32)), pa.array(paths, pa.string()),
                pa.array(start), pa.array(stop)]
        for ax in range(3):
            cols += [pa.array(bmin[:, ax]), pa.array(bmax[:, ax])]
        pq.write_table(pa.Table.from_arrays(cols, schema=CATALOG_INDEX_SCHEMA), tmp / INDEX_FILENAME, **_PQ_OPTS)
    entry = {"dir": _rel_dir(name, tag), "layout": "chunks", "kind": "catalog",
             "n_chunks": n_chunks, "n_side": int(n_side), "n_rows": int(stop[-1]) if n_chunks else 0}
    _merge_layout(store_root, name, tag, entry)
    return entry


def _relative_native(native_path: Path, store_root: Path) -> str:
    """Native path relative to the store root (resolved against it on read)."""
    native_path = Path(os.path.abspath(native_path))
    store_root = Path(os.path.abspath(store_root))
    try:
        return os.path.relpath(native_path, store_root)
    except ValueError:  # different drive
        return str(native_path)


def write_reference_catalog(store_root, name: str, z: float, native_path, adapter: NativeReaderAdapter,
                            *, box_size: float, n_side: int) -> dict:
    """Index-only catalog product over a chunk-sorted native file.

    Needs an adapter exposing ``chunk_ranges(path, z)`` (``packed_npy``).
    Bounding boxes are the chunk cells (the native file is not scanned).
    """
    if not hasattr(adapter, "chunk_ranges"):
        raise ValidationError(
            f"adapter {adapter.format!r} has no chunk ranges; a reference catalog needs 'packed_npy'")
    hdr = adapter.header(native_path)
    if int(hdr.get("n_side", -1)) != int(n_side):
        raise ValidationError(f"native header n_side={hdr.get('n_side')} differs from n_side={n_side}")
    store_root, tag = Path(store_root), ztag(z)
    ranges = adapter.chunk_ranges(native_path, z)
    n_chunks = int(n_side) ** 3
    n_rows = adapter.num_rows(native_path)
    start = np.zeros(n_chunks, dtype=np.int64)
    stop = np.zeros(n_chunks, dtype=np.int64)
    prev = 0
    for c in range(n_chunks):
        a, b = ranges.get(c, (prev, prev))
        if a != prev or b < a:
            raise FormatError(f"native chunk ranges of {native_path} are not contiguous at chunk {c}")
        start[c], stop[c], prev = a, b, b
    if prev != n_rows or any(not 0 <= c < n_chunks for c in ranges):
        raise FormatError(f"native chunk ranges of {native_path} do not cover its {n_rows} rows")
    rel = _relative_native(Path(native_path), store_root)
    bbox = [chunk_bbox(c, box_size, n_side) for c in range(n_chunks)]
    cols = [pa.array(np.arange(n_chunks, dtype=np.int32)),
            pa.array([rel if stop[c] > start[c] else "" for c in range(n_chunks)], pa.string()),
            pa.array(start), pa.array(stop)]
    for ax in range(3):
        cols += [pa.array([b[0][ax] for b in bbox]), pa.array([b[1][ax] for b in bbox])]
    with _staged_product(store_root, name, tag) as tmp:
        pq.write_table(pa.Table.from_arrays(cols, schema=CATALOG_INDEX_SCHEMA), tmp / INDEX_FILENAME, **_PQ_OPTS)
    entry = {"dir": _rel_dir(name, tag), "layout": "reference", "kind": "catalog",
             "n_chunks": n_chunks, "n_side": int(n_side), "n_rows": int(n_rows),
             "native": rel, "adapter": adapter.format}
    _merge_layout(store_root, name, tag, entry)
    return entry


# ---------------------------------------------------------------------- field

def _tile_grid(n_grid: int, tile_cells: int):
    if int(tile_cells) != tile_cells or tile_cells < 1:
        raise ValidationError(f"tile_cells must be a positive integer, got {tile_cells}")
    nt = math.ceil(n_grid / tile_cells)
    for a in range(nt):
        for b in range(nt):
            for c in range(nt):
                yield ((a * nt + b) * nt + c,
                       tuple(s for i in (a, b, c)
                             for s in (i * tile_cells, min(n_grid, (i + 1) * tile_cells))))


def _check_cube(shape) -> int:
    if len(shape) != 3 or not shape[0] == shape[1] == shape[2]:
        raise ValidationError(f"a field must be a cubic 3-D grid, got shape {tuple(shape)}")
    return int(shape[0])


def _field_index(tiles: list[tuple], paths: bool) -> pa.Table:
    cols = [pa.array([t[0] for t in tiles], pa.int32()),
            pa.array([f"tile_{t[0]}.npy" if paths else "" for t in tiles], pa.string())]
    for j in range(6):
        cols.append(pa.array([t[1][j] for t in tiles], pa.int32()))
    return pa.Table.from_arrays(cols, schema=FIELD_INDEX_SCHEMA)


def write_field_product(store_root, name: str, z: float, field, *, tile_cells: int = 32,
                        plan: ExecutionPlan | None = None) -> dict:
    """Write a cubic grid as ``tile_<id>.npy`` files, one slice of ``field`` at a time.

    ``field`` may be a memmap; it is never materialised whole.  Edge tiles are
    smaller when ``n_grid % tile_cells != 0``.  Returns the layout entry.
    """
    store_root, tag = Path(store_root), ztag(z)
    n_grid = _check_cube(field.shape)
    tiles = list(_tile_grid(n_grid, tile_cells))
    with _staged_product(store_root, name, tag) as tmp:
        for tid, (i0, i1, j0, j1, k0, k1) in tiles:
            np.save(tmp / f"tile_{tid}.npy",
                    np.ascontiguousarray(field[i0:i1, j0:j1, k0:k1]))
        pq.write_table(_field_index(tiles, True), tmp / INDEX_FILENAME, **_PQ_OPTS)
    entry = {"dir": _rel_dir(name, tag), "layout": "tiles", "kind": "field",
             "n_tiles": len(tiles), "n_grid": n_grid, "tile_cells": int(tile_cells)}
    _merge_layout(store_root, name, tag, entry)
    return entry


def write_reference_field(store_root, name: str, z: float, native_path, adapter: NativeReaderAdapter,
                          *, tile_cells: int = 32) -> dict:
    """Index-only field product: tile index from the native grid shape."""
    store_root, tag = Path(store_root), ztag(z)
    n_grid = _check_cube(adapter.field_shape(native_path))
    tiles = list(_tile_grid(n_grid, tile_cells))
    rel = _relative_native(Path(native_path), store_root)
    with _staged_product(store_root, name, tag) as tmp:
        pq.write_table(_field_index(tiles, False), tmp / INDEX_FILENAME, **_PQ_OPTS)
    entry = {"dir": _rel_dir(name, tag), "layout": "reference", "kind": "field",
             "n_tiles": len(tiles), "n_grid": n_grid, "tile_cells": int(tile_cells),
             "native": rel, "adapter": adapter.format}
    _merge_layout(store_root, name, tag, entry)
    return entry


# ------------------------------------------------------------------ lightcone

def write_lightcone_product(store_root, name: str, batches: Iterable[pa.RecordBatch], *,
                            nside_part: int, plan: ExecutionPlan | None = None, z: float = 0.0) -> dict:
    """Partition ``batches`` (columns ``ra``, ``dec`` in degrees) by NEST pixel of ``nside_part``."""
    plan = plan or ExecutionPlan()
    store_root, tag = Path(store_root), ztag(z)
    n_pix = 12 * int(nside_part) ** 2
    counts = np.zeros(n_pix, dtype=np.int64)

    def key_fn(batch: pa.RecordBatch) -> np.ndarray:
        try:
            ra = batch.column("ra").to_numpy(zero_copy_only=False)
            dec = batch.column("dec").to_numpy(zero_copy_only=False)
        except KeyError:
            raise ValidationError(
                f"lightcone batches need columns ra, dec; got {batch.schema.names}") from None
        return radec_to_pix(nside_part, ra, dec)

    def observe(batch: pa.RecordBatch, keys: np.ndarray) -> None:
        counts[:] += np.bincount(keys, minlength=n_pix)

    with _staged_product(store_root, name, tag) as tmp:
        stage = tmp / "_stage"
        stage.mkdir()
        _partition_write(batches, key_fn, 0, n_pix, lambda k: tmp / f"pix_{k}.parquet", stage,
                         lambda s: max(1, plan.batch_for(_row_bytes(s))), observe=observe)
        shutil.rmtree(stage)
        pix = np.flatnonzero(counts).astype(np.int32)
        table = pa.table({"pix": pa.array(pix), "path": pa.array([f"pix_{p}.parquet" for p in pix], pa.string()),
                          "n_rows": pa.array(counts[pix])}, schema=SKY_INDEX_SCHEMA)
        pq.write_table(table, tmp / INDEX_FILENAME, **_PQ_OPTS)
    entry = {"dir": _rel_dir(name, tag), "layout": "sky", "kind": "lightcone", "n_pix": len(pix),
             "nside_part": int(nside_part), "n_rows": int(counts.sum())}
    _merge_layout(store_root, name, tag, entry)
    return entry


# --------------------------------------------------------------- field ingest

def ingest_field(store_root, name: str, z: float, field, *, tile_cells: int = 32,
                 plan: ExecutionPlan | None = None) -> None:
    """Add a field product (tiles, layout entry, manifest declaration) to an existing store.

    Tiles are staged and renamed into place, then ``layout.json`` and
    ``manifest.json`` are rewritten atomically (each file is atomic; a crash
    between the two leaves a layout entry without declaration, which a retry
    reports as ``FormatError``).  An existing product at ``z`` is a ``FormatError``.
    """
    store_root, tag = Path(store_root), ztag(z)
    mpath = store_root / MANIFEST_FILENAME
    manifest = read_manifest(mpath)
    n_grid = _check_cube(field.shape)
    _check_new(store_root, name, tag)
    decls = list(manifest.products)
    existing = next((d for d in decls if d.name == name), None)
    if existing is not None and (existing.kind != "field" or existing.n_grid != n_grid):
        raise FormatError(f"product {name!r} exists as {existing.kind} (n_grid={existing.n_grid}); "
                          f"cannot add a field of n_grid={n_grid}")
    write_field_product(store_root, name, z, field, tile_cells=tile_cells, plan=plan)
    from oneuniverse.simulation.specs import ProductDecl
    if existing is None:
        decls.append(ProductDecl(name, "field", (float(z),), n_grid=n_grid))
    else:
        decls[decls.index(existing)] = replace(existing, redshifts=existing.redshifts + (float(z),))
    write_manifest(mpath, replace(manifest, products=tuple(decls)))
