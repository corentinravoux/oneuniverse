"""Read-side view over an OUF 3.0 dataset with vectorised partition pruning."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.geometry import Cone, SkyPatch, ZRange
from oneuniverse._core.healpix import cone_pixels, patch_pixels
from oneuniverse.data.manifest import (DATA_DIR, DATASET_UUID_KEY, INDEX_FILENAME,
                                       MANIFEST_FILENAME, Manifest, read_manifest,
                                       read_partition_index)
from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift
from oneuniverse.data.schema import Geometry

_RAD = np.pi / 180.0


def _rad(x) -> pc.Expression:
    return x * _RAD


def cone_expr(cone: Cone) -> pc.Expression:
    """Closed haversine cone membership in float64, mirroring ``Cone.contains``.

    Same operation order as numpy. Exact agreement on points constructed *on* the
    boundary is not achievable: Arrow's libm ``sin``/``pow`` and numpy's (possibly
    SIMD) ``sin`` can differ by an ulp, so ~4e-5 of such adversarial points flip
    (the property test asserts >= 99.99 % agreement). Generic points never differ.
    """
    ra = _rad(pc.field("ra").cast(pa.float64()))
    dec = _rad(pc.field("dec").cast(pa.float64()))
    ra0, dec0 = float(np.radians(cone.ra)), float(np.radians(cone.dec))
    sd = pc.sin((dec - dec0) / 2.0)
    sr = pc.sin((ra - ra0) / 2.0)
    # Same operation order as numpy in Cone.contains: sd**2 + (cos*cos0) * sr**2.
    # The squares use pc.power: Acero re-associates ``c * (s * s)`` into ``(c * s) * s``,
    # which differs from numpy by an ulp and flips boundary points.
    a = sd * sd + (pc.cos(dec) * float(np.cos(dec0))) * pc.power(sr, 2.0)
    a = pc.max_element_wise(pc.min_element_wise(a, 1.0), 0.0)
    sep = pc.asin(pc.sqrt(a)) * 2.0
    return sep <= float(np.radians(cone.radius))


def patch_expr(patch: SkyPatch) -> pc.Expression:
    """Closed RA/Dec box membership; RA taken modulo 360, wrap-aware."""
    ra_in = pc.field("ra").cast(pa.float64())
    ra = ra_in - pc.floor(ra_in / 360.0) * 360.0
    if patch.wraps:
        in_ra = (ra >= patch.ra_min) | (ra <= patch.ra_max)
    else:
        in_ra = (ra >= patch.ra_min) & (ra <= patch.ra_max)
        if patch.ra_max == 360.0:
            in_ra = in_ra | (ra_in == 360.0)
    dec = pc.field("dec")
    return in_ra & (dec >= patch.dec_min) & (dec <= patch.dec_max)


def range_expr(column: str, lo: float, hi: float) -> pc.Expression:
    """Closed interval ``lo <= column <= hi``."""
    f = pc.field(column)
    return (f >= lo) & (f <= hi)


def _interval(v, what: str) -> tuple[float, float]:
    if isinstance(v, ZRange):
        return v.lo, v.hi
    try:
        lo, hi = v
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{what} must be a ZRange or a (lo, hi) pair, got {v!r}") from e
    if not lo <= hi:
        raise ValidationError(f"{what} requires lo <= hi, got ({lo}, {hi})")
    return lo, hi


class DatasetView:
    """Lazy handle on an OUF 3.0 dataset: manifest and index read once, one
    pyarrow dataset, row-group-free partition pruning from ``_index.parquet``."""

    def __init__(self, path: Path, manifest: Manifest, index: pa.Table) -> None:
        self.path = path
        if int(pc.sum(index["n_rows"]).as_py() or 0) != manifest.n_rows:
            raise FormatError(
                f"{path}: partition index rows ({pc.sum(index['n_rows']).as_py()}) do not "
                f"match manifest n_rows ({manifest.n_rows})")
        self.manifest = manifest
        self.index = index
        self._dataset = pads.dataset(
            path / DATA_DIR, format="parquet",
            partitioning=pads.partitioning(pa.schema([("cell", pa.int32())]),
                                           flavor="hive"))
        self._schema = manifest.schema
        self._cells = index["cell"].to_numpy().astype(np.int32)
        self._paths = index["path"].to_pylist()

    @classmethod
    def from_path(cls, path: str | Path) -> DatasetView:
        path = Path(path)
        if not (path / MANIFEST_FILENAME).exists():
            raise FormatError(f"{path} is not an OUF dataset (no {MANIFEST_FILENAME})")
        manifest = read_manifest(path / MANIFEST_FILENAME)
        try:
            index = read_partition_index(path / INDEX_FILENAME)
        except (OSError, pa.ArrowException) as e:
            raise FormatError(f"{path}: cannot read {INDEX_FILENAME}: {e}") from e
        return cls(path, manifest, index)

    # ---- properties -------------------------------------------------
    @property
    def geometry(self) -> Geometry:
        return self.manifest.geometry

    @property
    def n_rows(self) -> int:
        return self.manifest.n_rows

    @property
    def n_partitions(self) -> int:
        return self.index.num_rows

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def columns(self) -> list[str]:
        return list(self._schema.names)

    @property
    def nside(self) -> int:
        return self.manifest.partitioning.nside

    @property
    def version(self) -> tuple[int, str, str]:
        """``(validity.version, created_utc, dataset_uuid)`` of the opened publish."""
        m = self.manifest
        return (m.validity.version, m.created_utc, m.dataset_uuid)

    # ---- stale-view guard -------------------------------------------
    def _stale(self) -> FormatError:
        return FormatError(
            f"dataset {self.path} was rewritten since this view was opened; reopen it")

    def _check_token(self, md: pq.FileMetaData) -> None:
        kv = md.metadata or {}
        if kv.get(DATASET_UUID_KEY.encode(), b"").decode() != self.manifest.dataset_uuid:
            raise self._stale()

    def _open_file(self, partition: int) -> pq.ParquetFile:
        try:
            pf = pq.ParquetFile(self.path / self._paths[partition])
        except (OSError, pa.ArrowException) as e:
            raise self._stale() from e
        self._check_token(pf.metadata)
        return pf

    def _verify(self, partitions: np.ndarray) -> None:
        """Check the dataset token of each file (footer read only; never cached, so a
        rewrite between two scans is caught)."""
        for i in partitions.tolist():
            self._open_file(i)

    # ---- pruning ----------------------------------------------------
    def _time_column(self) -> str:
        ext = self.manifest.extension("temporal")
        if ext is None or "time_column" not in ext:
            raise FormatError(
                "time selection needs the 'temporal' extension with a 'time_column'")
        return ext["time_column"]

    def _intervals(self, z, t, ranges) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        if z is not None:
            out["z"] = _interval(z, "z")
        if t is not None:
            out[self._time_column()] = _interval(t, "t")
        for k, v in (ranges or {}).items():
            if k not in self._schema.names:
                raise ValidationError(f"ranges column {k!r} not in dataset schema")
            out[k] = _interval(v, f"ranges[{k!r}]")
        return out

    def _stat(self, name: str) -> np.ndarray:
        col = self.index[name].cast(pa.float64()).fill_null(float("nan"))
        return col.to_numpy(zero_copy_only=False)

    def _mask(self, cone, patch, z, t, ranges) -> np.ndarray:
        keep = np.ones(self.n_partitions, bool)
        for col, (lo, hi) in self._intervals(z, t, ranges).items():
            if f"{col}_min" in self.index.schema.names and f"{col}_max" in self.index.schema.names:
                # NaN stats compare False, so unknown partitions are never pruned.
                keep &= ~(self._stat(f"{col}_max") < lo) & ~(self._stat(f"{col}_min") > hi)
        sky = None
        if cone is not None:
            sky = cone_pixels(self.nside, cone.ra, cone.dec, cone.radius,
                              nest=self.manifest.partitioning.nest)
        if patch is not None:
            p = patch_pixels(self.nside, patch.ra_min, patch.ra_max,
                             patch.dec_min, patch.dec_max,
                             nest=self.manifest.partitioning.nest)
            sky = p if sky is None else np.intersect1d(sky, p)
        if sky is not None:
            keep &= np.isin(self._cells, sky)
        return keep

    def cells_for(self, *, cone=None, patch=None, z=None, t=None,
                  ranges: Mapping[str, tuple[float, float]] | None = None) -> np.ndarray:
        """Sorted unique int32 cells holding partitions that may match."""
        keep = self._mask(cone, patch, z, t, ranges)
        return np.unique(self._cells[keep])

    # ---- scanning ---------------------------------------------------
    def _columns(self, columns) -> list[str]:
        cols = self.columns if columns is None else list(columns)
        for c in cols:
            if c not in self._schema.names:
                raise ValidationError(f"column {c!r} not in dataset schema")
        return cols

    def _empty(self, cols: list[str]) -> pa.Table:
        return pa.schema([self._schema.field(c) for c in cols]).empty_table()

    def scan(self, *, columns=None, cone=None, patch=None, z=None, t=None,
             ranges=None, filter: pc.Expression | None = None,
             batch_size: int = 65_536) -> Iterator[pa.RecordBatch]:
        cols = self._columns(columns)
        keep = self._mask(cone, patch, z, t, ranges)
        self._verify(np.flatnonzero(keep))
        cells = np.unique(self._cells[keep])
        if len(cells) == 0:
            return iter(())
        expr = pc.field("cell").isin(pa.array(cells, pa.int32()))
        if cone is not None:
            expr = expr & cone_expr(cone)
        if patch is not None:
            expr = expr & patch_expr(patch)
        for col, (lo, hi) in self._intervals(z, t, ranges).items():
            expr = expr & range_expr(col, lo, hi)
        if filter is not None:
            expr = expr & filter
        scanner = self._dataset.scanner(columns=cols, filter=expr, batch_size=batch_size)
        return scanner.to_batches()

    def table(self, *, columns=None, **sel) -> pa.Table:
        cols = self._columns(columns)
        batches = list(self.scan(columns=cols, **sel))
        if not batches:
            return self._empty(cols)
        return pa.Table.from_batches(batches)

    def read(self, *, columns=None, **sel) -> pd.DataFrame:
        t = self.table(columns=columns, **sel)
        cats = [f.name for f in t.schema if pa.types.is_dictionary(f.type)]
        return t.to_pandas(categories=cats)

    # ---- random access ----------------------------------------------
    def _read_file(self, partition: int, columns: list[str]) -> pa.Table:
        return self._open_file(partition).read(columns=columns)

    def iter_partitions(self, columns=None) -> Iterator[tuple[int, pa.Table]]:
        cols = self._columns(columns)
        for i in range(self.n_partitions):
            yield i, self._read_file(i, cols)

    def _take_from_file(self, partition: int, offsets: np.ndarray, cols: list[str]) -> pa.Table:
        """Rows at *offsets* of one file, reading only the row groups that hold them."""
        pf = self._open_file(partition)
        md = pf.metadata
        sizes = np.array([md.row_group(g).num_rows for g in range(md.num_row_groups)], np.int64)
        ends = np.cumsum(sizes)
        rg = np.searchsorted(ends, offsets, side="right")
        sel = np.unique(rg)
        base = np.concatenate([[0], np.cumsum(sizes[sel])[:-1]])
        local = offsets - (ends[rg] - sizes[rg]) + base[np.searchsorted(sel, rg)]
        tab = pf.read_row_groups(sel.tolist(), columns=cols)
        return tab.take(pa.array(local))

    def take(self, partition, offset, *, columns=None) -> pa.Table:
        """Rows ``(partition, offset)`` in the order given; partition = row in ``.index``."""
        cols = self._columns(columns)
        part, off = np.asarray(partition).ravel(), np.asarray(offset).ravel()
        for name, a in (("partition", part), ("offset", off)):
            if a.size and a.dtype.kind not in "iu":
                raise ValidationError(f"{name} must be an integer array, got dtype {a.dtype}")
        part, off = part.astype(np.int64, copy=False), off.astype(np.int64, copy=False)
        if part.shape != off.shape:
            raise ValidationError("partition and offset must have the same length")
        if part.size == 0:
            return self._empty(cols)
        if part.min() < 0 or part.max() >= self.n_partitions:
            raise ValidationError("partition id out of range of the index")
        n_rows = self.index["n_rows"].to_numpy()
        if off.min() < 0 or np.any(off >= n_rows[part]):
            raise ValidationError("offset out of range of its partition")
        order = np.argsort(part, kind="stable")
        pieces, sorted_part = [], part[order]
        bounds = np.flatnonzero(np.diff(sorted_part)) + 1
        for grp in np.split(np.arange(part.size), bounds):
            idx = order[grp]
            pieces.append(self._take_from_file(int(part[idx[0]]), off[idx], cols))
        out = pa.concat_tables(pieces)
        inverse = np.empty(part.size, np.int64)
        inverse[order] = np.arange(part.size)
        return out.take(pa.array(inverse))

    def load_pdf(self, rows: pa.Table | None = None) -> ProbabilisticRedshift:
        """PDFs of *rows* (a table holding the value column) or of the whole dataset."""
        ext = self.manifest.extension("pdf")
        if ext is None:
            raise FormatError("dataset has no 'pdf' extension")
        spec = PdfSpec.from_dict(ext)
        if rows is None:
            rows = self.table(columns=[spec.value_column])
        return ProbabilisticRedshift.from_table(rows, spec)
