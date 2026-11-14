"""Read side of the store: index access and selector reads (box, cone, field box).

Every reader takes a selector; the only whole-product read is :meth:`SimStore.scan_full`.
Catalog positions are stored wrapped into ``[0, box_size]``; a cube that crosses
the box boundary is split (:func:`split_periodic`) and rows are returned with
their stored (wrapped) positions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from oneuniverse._core.errors import (
    FormatError,
    NotFoundError,
    ValidationError,
)
from oneuniverse._core.geometry import Cone, Cube
from oneuniverse._core.healpix import cone_pixels
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    OUFSimManifest,
    read_manifest,
)
from oneuniverse.simulation.store.index import overlapping_chunks, split_periodic
from oneuniverse.simulation.store.native import get_adapter

INDEX_FILENAME = "_index.parquet"
_POS = ("x", "y", "z")
_HINT = {"catalog": "read_box / iter_box / scan_full", "field": "read_field_box",
         "lightcone": "read_cone"}


def _row_bytes(schema: pa.Schema) -> int:
    total = 0
    for f in schema:
        try:
            total += max(1, f.type.bit_width // 8)
        except ValueError:
            total += 16
    return max(total, 1)


def _tag(z: float) -> str:
    return f"{float(z):.3f}"


class SimStore:
    """A partial-access simulation store rooted at ``root``."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        if not (self.root / MANIFEST_FILENAME).exists():
            raise NotFoundError(f"no {MANIFEST_FILENAME} in {self.root}")
        self.manifest: OUFSimManifest = read_manifest(self.root / MANIFEST_FILENAME)
        try:
            self.layout: dict = json.loads((self.root / LAYOUT_FILENAME).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FormatError(f"cannot read {LAYOUT_FILENAME} in {self.root}: {exc}") from exc
        self._index_cache: dict[tuple[str, str], pa.Table] = {}
        self.last_read_stats: dict = {}

    @property
    def products(self) -> tuple[str, ...]:
        """Committed products: declared in the manifest and listed in ``layout.json``."""
        committed = self.layout.get("products", {})
        return tuple(p.name for p in self.manifest.products if committed.get(p.name))

    def _entry(self, product: str, z: float) -> dict:
        try:
            return self.layout["products"][product][_tag(z)]
        except KeyError:
            raise NotFoundError(f"no product {product!r} at z={_tag(z)} in {self.root}") from None

    def redshifts(self, product: str) -> tuple[float, ...]:
        zs = self.manifest.product(product).redshifts
        have = self.layout["products"].get(product, {})
        return tuple(z for z in zs if _tag(z) in have)

    def index(self, product: str, z: float) -> pa.Table:
        key = (product, _tag(z))
        if key not in self._index_cache:
            entry = self._entry(product, z)
            self._index_cache[key] = pq.read_table(self.root / entry["dir"] / INDEX_FILENAME)
        return self._index_cache[key]

    # ------------------------------------------------------------- helpers

    def _typed_entry(self, product: str, z: float, kind: str, method: str) -> dict:
        entry = self._entry(product, z)
        if entry.get("kind") != kind:
            raise ValidationError(
                f"product {product!r} is a {entry.get('kind')} product; use {_HINT[entry['kind']]}"
                f" (not {method})")
        return entry

    def _box_size(self) -> float:
        return float(self.manifest.box_size)

    def _chunk_rows(self, entry: dict, product: str, z: float, cid: int,
                    columns: list[str] | None) -> pa.Table:
        """Rows of chunk ``cid`` (reencode: its parquet file; reference: a native row range)."""
        row = self.index(product, z).slice(cid, 1).to_pylist()[0]
        if entry["layout"] == "reference":
            adapter = get_adapter(entry["adapter"])
            return adapter.read_rows(self.root / row["path"], row["row_start"], row["row_stop"],
                                     columns=columns)
        return pq.read_table(self.root / entry["dir"] / row["path"], columns=columns)

    def _check_columns(self, entry: dict, product: str, z: float, columns, kind: str) -> None:
        if columns is None:
            return
        names = self._schema_names(entry, product, z, kind)
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValidationError(f"column(s) {missing} not in schema of product {product!r}; available: {names}")

    def _schema_names(self, entry: dict, product: str, z: float, kind: str) -> list[str]:
        return self._schema(entry, product, z, kind).names

    def _schema(self, entry: dict, product: str, z: float, kind: str) -> pa.Schema:
        index = self.index(product, z)
        if kind == "lightcone":
            paths = [p for p in index.column("path").to_pylist() if p]
            return pq.read_schema(self.root / entry["dir"] / paths[0]) if paths else pa.schema([])
        for row in index.to_pylist():
            if row["row_stop"] > row["row_start"] and row["path"]:
                if entry["layout"] == "reference":
                    return get_adapter(entry["adapter"]).read_rows(
                        self.root / row["path"], row["row_start"], row["row_start"] + 1).schema
                return pq.read_schema(self.root / entry["dir"] / row["path"])
        return pa.schema([])

    def _box_chunks(self, entry: dict, product: str, z: float, cube: Cube,
                    plan: ExecutionPlan) -> tuple[list[int], int]:
        index = self.index(product, z)
        ids = overlapping_chunks(index, cube, self._box_size())
        rows = index.column("row_stop").to_numpy() - index.column("row_start").to_numpy()
        ids = [int(c) for c in ids if rows[c] > 0]
        if plan.mode is ExecutionMode.MPI:
            comm = plan.mpi_comm
            ids = ids[comm.Get_rank()::comm.Get_size()]
        return ids, index.num_rows

    def _filter_chunk(self, entry, product, z, cid, pieces, need) -> pa.Table:
        t = self._chunk_rows(entry, product, z, cid, need)
        x, y, zz = (t.column(a).to_numpy(zero_copy_only=False) for a in _POS)
        mask = np.zeros(t.num_rows, dtype=bool)
        for piece in pieces:
            mask |= piece.contains(x, y, zz)
        return t.filter(pa.array(mask))

    def _prepare_box(self, product, z, cube, columns, plan, method):
        entry = self._typed_entry(product, z, "catalog", method)
        plan = plan or ExecutionPlan()
        self._check_columns(entry, product, z, columns, "catalog")
        need = None if columns is None else list(dict.fromkeys([*columns, *_POS]))
        ids, total = self._box_chunks(entry, product, z, cube, plan)
        pieces = split_periodic(cube, self._box_size())
        return entry, plan, need, ids, total, pieces

    # --------------------------------------------------------------- reads

    def read_box(self, product: str, z: float, cube: Cube, *, columns: Sequence[str] | None = None,
                 plan: ExecutionPlan | None = None) -> pa.Table:
        """Rows of a catalog product inside ``cube`` (closed, periodic).

        The cube is split at the box faces, only chunks whose bounding box meets
        a piece are read (threaded under ``plan.n_threads``, concatenated in
        ascending chunk id), and an exact filter is applied to the stored
        positions, which are returned as stored (wrapped, not unwrapped).
        ``columns`` is applied after filtering.  With ``ExecutionMode.MPI`` the
        chunk ids are partitioned by rank (``ids[rank::size]``) and each rank
        returns only its own rows; nothing is gathered.  ``last_read_stats``
        records ``chunks_total``, ``chunks_read``, ``rows`` and ``bytes``.
        """
        columns = None if columns is None else list(columns)
        entry, plan, need, ids, total, pieces = self._prepare_box(
            product, z, cube, columns, plan, "read_box")
        with ThreadPoolExecutor(max_workers=plan.n_threads) as pool:
            tables = list(pool.map(
                lambda c: self._filter_chunk(entry, product, z, c, pieces, need), ids))
        if tables:
            out = pa.concat_tables(tables)
        else:
            out = self._schema(entry, product, z, "catalog").empty_table()
        if columns is not None:
            out = out.select(columns)
        self.last_read_stats = {"chunks_total": total, "chunks_read": len(ids),
                                "rows": out.num_rows, "bytes": out.nbytes}
        return out

    def iter_box(self, product: str, z: float, cube: Cube, *, columns: Sequence[str] | None = None,
                 plan: ExecutionPlan | None = None) -> Iterator[pa.Table]:
        """Like :meth:`read_box`, one chunk at a time, in batches of at most ``plan.batch_for`` rows.

        The batch size is set by the width of the columns actually returned
        (safety 0.5 of ``plan.memory_budget_bytes``).  MPI mode partitions chunks as in
        :meth:`read_box`.  ``last_read_stats`` is updated as chunks are consumed.
        """
        columns = None if columns is None else list(columns)
        entry, plan, need, ids, total, pieces = self._prepare_box(
            product, z, cube, columns, plan, "iter_box")
        return self._iter_chunks(entry, product, z, ids, total, pieces, need, columns, plan)

    def _iter_chunks(self, entry, product, z, ids, total, pieces, need, columns, plan):
        stats = {"chunks_total": total, "chunks_read": 0, "rows": 0, "bytes": 0}
        self.last_read_stats = stats
        for cid in ids:
            t = self._filter_chunk(entry, product, z, cid, pieces, need) if pieces else \
                self._chunk_rows(entry, product, z, cid, need)
            if columns is not None:
                t = t.select(columns)
            stats["chunks_read"] += 1
            step = plan.batch_for(_row_bytes(t.schema))
            for s in range(0, t.num_rows, step):
                b = t.slice(s, step)
                stats["rows"] += b.num_rows
                stats["bytes"] += b.nbytes
                yield b

    def scan_full(self, product: str, z: float, *, columns: Sequence[str] | None = None,
                  plan: ExecutionPlan | None = None) -> Iterator[pa.Table]:
        """Stream the whole catalog product at ``z``, chunk by chunk, in chunk-id order.

        This is the only reader without a selector: it walks every chunk of the
        whole product, so on a large simulation it reads the entire dataset.
        Prefer :meth:`read_box` / :meth:`iter_box`.  Batches obey ``plan.batch_for``.
        """
        columns = None if columns is None else list(columns)
        entry = self._typed_entry(product, z, "catalog", "scan_full")
        plan = plan or ExecutionPlan()
        self._check_columns(entry, product, z, columns, "catalog")
        index = self.index(product, z)
        rows = index.column("row_stop").to_numpy() - index.column("row_start").to_numpy()
        ids = [int(c) for c in index.column("chunk_id").to_numpy() if rows[c] > 0]
        return self._iter_chunks(entry, product, z, ids, index.num_rows, [], columns, columns, plan)

    def read_field_box(self, name: str, z: float, cube: Cube) -> np.ndarray:
        """Sub-grid (float32) of every cell intersecting ``cube``, periodic.

        Cell ``i`` covers ``[i*cell, (i+1)*cell)``; indices ``floor(lo/cell) ..
        ceil(hi/cell)-1`` per axis are taken modulo ``n_grid`` (wrapped order, at
        most ``n_grid`` cells per axis).  Only intersecting tiles are read.
        """
        entry = self._typed_entry(name, z, "field", "read_field_box")
        n_grid, cell = int(entry["n_grid"]), self._box_size() / int(entry["n_grid"])
        axes = []
        for axis, (lo, hi) in enumerate(zip(cube.lo, cube.hi)):
            if hi - lo > self._box_size() * (1 + 1e-12):
                raise ValidationError(
                    f"cube side {hi - lo} on axis {axis} exceeds box_size {self._box_size()}")
            a = int(np.floor(lo / cell))
            b = max(int(np.ceil(hi / cell)), a + 1)
            axes.append(np.arange(a, min(b, a + n_grid)) % n_grid)
        out = np.empty(tuple(len(a) for a in axes), dtype=np.float32)
        index = self.index(name, z)
        for row in index.to_pylist():
            lo = (row["i0"], row["j0"], row["k0"])
            hi = (row["i1"], row["j1"], row["k1"])
            sel = [np.flatnonzero((ax >= l) & (ax < h)) for ax, l, h in zip(axes, lo, hi)]
            if any(len(s) == 0 for s in sel):
                continue
            local = [ax[s] - l for ax, s, l in zip(axes, sel, lo)]
            if entry["layout"] == "reference":
                tile = get_adapter(entry["adapter"]).read_field_region(
                    self.root / entry["native"], lo, hi)
            else:
                tile = np.load(self.root / entry["dir"] / row["path"], mmap_mode="r")
            out[np.ix_(*sel)] = np.asarray(tile)[np.ix_(*local)]
        return out

    def read_cone(self, product: str, cone: Cone, *, z_range: tuple[float, float] | None = None,
                  columns: Sequence[str] | None = None) -> pa.Table:
        """Rows of a lightcone product inside ``cone`` (and ``z_range``, closed, on column ``z``).

        Pixels are pruned with the HEALPix cover of the cone at the stored
        ``nside_part``; an exact angular filter follows.  Lightcone products only.
        """
        columns = None if columns is None else list(columns)
        z0 = self._lightcone_z(product)
        entry = self._typed_entry(product, z0, "lightcone", "read_cone")
        self._check_columns(entry, product, z0, columns, "lightcone")
        if z_range is not None and not z_range[0] <= z_range[1]:
            raise ValidationError(f"z_range must be (zmin, zmax) with zmin <= zmax, got {z_range}")
        index = self.index(product, z0)
        cover = cone_pixels(int(entry["nside_part"]), cone.ra, cone.dec, cone.radius)
        sel = index.filter(pc.is_in(index["pix"], pa.array(cover, pa.int32())))
        need = None if columns is None else list(dict.fromkeys(
            [*columns, "ra", "dec", *(["z"] if z_range else [])]))
        tables = []
        for path in sel.column("path").to_pylist():
            t = pq.read_table(self.root / entry["dir"] / path, columns=need)
            mask = cone.contains(t["ra"].to_numpy(), t["dec"].to_numpy())
            if z_range is not None:
                zz = t["z"].to_numpy()
                mask &= (zz >= z_range[0]) & (zz <= z_range[1])
            tables.append(t.filter(pa.array(mask)))
        out = pa.concat_tables(tables) if tables else self._schema(entry, product, z0, "lightcone").empty_table()
        return out if columns is None else out.select(columns)

    def _lightcone_z(self, product: str) -> float:
        try:
            tags = self.layout["products"][product]
        except KeyError:
            raise NotFoundError(f"no product {product!r} in {self.root}") from None
        first = next(iter(tags), None)
        if first is None:
            raise NotFoundError(f"no product {product!r} in {self.root}")
        if tags[first].get("kind") != "lightcone":
            raise ValidationError(
                f"product {product!r} is a {tags[first].get('kind')} product; "
                f"use {_HINT[tags[first]['kind']]} (not read_cone)")
        return float(first)
