"""Streaming OUF 3.0 writer: record batches -> HEALPix-partitioned parquet.

Two passes over on-disk data, never over memory: batches are staged to
``<out>.tmp/stage`` (the row count, hence the partition ``nside``, is only known
at the end), then re-scanned and written hive-partitioned by ``cell``. The
whole dataset is published atomically with :func:`oneuniverse._core.fs.atomic_dir`.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, SchemaError, ValidationError
from oneuniverse._core.fs import atomic_dir, hash_file
from oneuniverse._core.healpix import NSIDE_FINE, radec_to_pix
from oneuniverse.data.manifest import (
    DATA_DIR,
    FORMAT_VERSION,
    INDEX_FILENAME,
    INDEX_SCHEMA,
    MANIFEST_FILENAME,
    LoaderInfo,
    Manifest,
    OriginalFile,
    Partitioning,
    SurveyMeta,
    Validity,
    read_manifest,
    write_manifest,
    write_partition_index,
)
from oneuniverse.data.schema import (
    Geometry,
    cast_to_core,
    core_schema,
    validate_schema,
    validate_z_type_values,
)

MIN_ROWS_PER_PARTITION = 5_000
MAX_ROWS_PER_FILE = 2_000_000
_STAGE = "stage"
_BUCKET_NSIDE = 8                       # stage buckets: NEST nside 8, <= 768 of them
_BUCKET_SHIFT = 2 * int(np.log2(NSIDE_FINE // _BUCKET_NSIDE))
_STAGE_BUFFER_BYTES = 48 * 2**20        # in-memory cap of the stage pass
_HISTORY = "_history"
_ROW_GROUP = 131_072
_STATS_BASE = ("ra", "dec", "z")


def auto_partition_nside(n_rows: int) -> int:
    """Finest power-of-2 nside <= 32 with >= MIN_ROWS_PER_PARTITION rows per cell."""
    nside = NSIDE_FINE
    while nside > 1 and n_rows / (12 * nside * nside) < MIN_ROWS_PER_PARTITION:
        nside //= 2
    return nside


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _as_batches(data: pa.Table | Iterable[pa.RecordBatch], batch_size: int
                ) -> Iterator[pa.RecordBatch]:
    if isinstance(data, pa.Table):
        yield from data.to_batches(max_chunksize=batch_size)
    else:
        yield from data


def _check_batch_schema(schema: pa.Schema, geometry: Geometry) -> None:
    probe = schema
    if "hpix32" not in probe.names:  # the writer supplies it
        probe = probe.append(pa.field("hpix32", pa.int32()))
    validate_schema(probe, geometry)


def _prepare(batch: pa.RecordBatch, geometry: Geometry, has_z_type: bool) -> pa.RecordBatch:
    table = pa.Table.from_batches([batch])
    if "hpix32" not in table.schema.names:
        hp = radec_to_pix(NSIDE_FINE, table["ra"].to_numpy(zero_copy_only=False),
                          table["dec"].to_numpy(zero_copy_only=False))
        table = table.append_column("hpix32", pa.array(hp, pa.int32()))
    if has_z_type:
        validate_z_type_values(table["z_type"])
    table = cast_to_core(table, geometry).combine_chunks()
    return table.to_batches()[0] if table.num_rows else pa.RecordBatch.from_pylist(
        [], schema=table.schema)


class _Counter:
    def __init__(self) -> None:
        self.n_rows = 0
        self.schema: pa.Schema | None = None


def _validated_stream(data, geometry: Geometry, batch_size: int, counter: _Counter
                      ) -> Iterator[pa.RecordBatch]:
    has_z_type = "z_type" in core_schema(geometry).names
    for raw in _as_batches(data, batch_size):
        if counter.schema is None:
            _check_batch_schema(raw.schema, geometry)
        if raw.num_rows == 0 and counter.schema is not None:
            continue
        batch = _prepare(raw, geometry, has_z_type)
        if counter.schema is None:
            counter.schema = batch.schema.remove_metadata()
        elif not batch.schema.remove_metadata().equals(counter.schema):
            raise SchemaError(
                "batch schema differs from the first batch:\n"
                f"first: {counter.schema}\nbatch: {batch.schema}")
        counter.n_rows += batch.num_rows
        if batch.num_rows:
            yield batch


def _hist_name(created_utc: str) -> str:
    return created_utc.replace(":", "-") + ".manifest.json"


def _round_out(vmin, vmax) -> tuple[float, float]:
    """float64 bounds that still contain int64 stats beyond 2**53 (round outwards)."""
    lo, hi = float(vmin), float(vmax)
    if lo > vmin:
        lo = float(np.nextafter(lo, -np.inf))
    if hi < vmax:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


def _column_minmax(md: pq.FileMetaData, names: Sequence[str]) -> dict[str, tuple[float | None, float | None]]:
    cols = {md.schema.column(j).path: j for j in range(md.num_columns)}
    out: dict[str, tuple[float | None, float | None]] = {}
    for name in names:
        j = cols[name]
        lo = hi = None
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(j).statistics
            if st is None or not st.has_min_max:
                continue
            a, b = _round_out(st.min, st.max)
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        out[name] = (lo, hi)
    return out


def _build_index(root: Path, stats_cols: Sequence[str]) -> pa.Table:
    names = list(_STATS_BASE) + list(stats_cols)
    rows: dict[str, list] = {f.name: [] for f in INDEX_SCHEMA}
    for c in stats_cols:
        rows[f"{c}_min"], rows[f"{c}_max"] = [], []
    files = sorted((root / DATA_DIR).rglob("*.parquet"))
    for f in files:
        md = pq.ParquetFile(f).metadata
        mm = _column_minmax(md, names)
        rows["path"].append(f.relative_to(root).as_posix())
        rows["cell"].append(int(f.parent.name.removeprefix("cell=")))
        rows["n_rows"].append(md.num_rows)
        for n in names:
            rows[f"{n}_min"].append(mm[n][0])
            rows[f"{n}_max"].append(mm[n][1])
    fields = list(INDEX_SCHEMA) + [
        pa.field(f"{c}_{s}", pa.float64()) for c in stats_cols for s in ("min", "max")]
    table = pa.table({f.name: pa.array(rows[f.name], f.type) for f in fields})
    return table.sort_by([("cell", "ascending"), ("path", "ascending")])


def _flush_bucket(stage: Path, bucket: int, tables: list[pa.Table], serial: list[int]) -> None:
    d = stage / f"{bucket:04d}"
    d.mkdir(exist_ok=True)
    table = pa.concat_tables(tables).combine_chunks()
    serial[0] += 1
    pq.write_table(table, d / f"s{serial[0]}.parquet", row_group_size=_ROW_GROUP,
                   compression="zstd")


def _stage_stream(stream: Iterator[pa.RecordBatch], stage: Path) -> None:
    """Spill *stream* into per-bucket parquet files with a bounded in-memory buffer."""
    buffers: dict[int, list[pa.Table]] = {}
    sizes: dict[int, int] = {}
    total = 0
    serial = [0]
    for batch in stream:
        hp = batch.column(batch.schema.get_field_index("hpix32")).to_numpy(zero_copy_only=False)
        bucket = hp.astype(np.int64) >> _BUCKET_SHIFT
        order = np.argsort(bucket, kind="stable")
        sb = bucket[order]
        uniq, start = np.unique(sb, return_index=True)
        stop = np.append(start[1:], len(sb))
        for b, a, z in zip(uniq.tolist(), start.tolist(), stop.tolist(), strict=True):
            piece = pa.Table.from_batches([batch]).take(pa.array(order[a:z]))
            buffers.setdefault(b, []).append(piece)
            sizes[b] = sizes.get(b, 0) + piece.nbytes
            total += piece.nbytes
        while total > _STAGE_BUFFER_BYTES:
            big = max(sizes, key=sizes.__getitem__)
            _flush_bucket(stage, big, buffers.pop(big), serial)
            total -= sizes.pop(big)
    for b in sorted(buffers):
        _flush_bucket(stage, b, buffers[b], serial)


def _read_bucket(stage: Path, bucket_dir: Path, schema: pa.Schema) -> pa.Table:
    parts = [pq.ParquetFile(f).read() for f in sorted(bucket_dir.glob("*.parquet"))]
    table = pa.concat_tables(parts).unify_dictionaries().combine_chunks()
    return table.take(pc.sort_indices(table["hpix32"]))


class _CellWriter:
    """Writes consecutive cells with a single open ParquetWriter and full row groups."""

    def __init__(self, root: Path, schema: pa.Schema) -> None:
        self.root, self.schema = root, schema
        self.cell: int | None = None
        self.writer: pq.ParquetWriter | None = None
        self.part = 0
        self.in_file = 0
        self.pending: list[pa.Table] = []
        self.pending_rows = 0

    def _open(self) -> None:
        d = self.root / DATA_DIR / f"cell={self.cell}"
        d.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(
            d / f"part-{self.part}.parquet", self.schema, compression="zstd",
            write_statistics=True, use_dictionary=True)
        self.in_file = 0

    def _write(self, table: pa.Table) -> None:
        file_rows = (MAX_ROWS_PER_FILE // _ROW_GROUP) * _ROW_GROUP
        pos = 0
        while pos < table.num_rows:
            if self.writer is not None and self.in_file >= file_rows:
                self.writer.close()
                self.writer = None
                self.part += 1
            if self.writer is None:
                self._open()
            n = min(table.num_rows - pos, file_rows - self.in_file)
            self.writer.write_table(table.slice(pos, n), row_group_size=_ROW_GROUP)
            self.in_file += n
            pos += n

    def _drain(self, final: bool) -> None:
        if not self.pending:
            return
        table = pa.concat_tables(self.pending).combine_chunks()
        k = table.num_rows if final else (table.num_rows // _ROW_GROUP) * _ROW_GROUP
        if k:
            self._write(table.slice(0, k))
        rest = table.slice(k)
        self.pending = [rest] if rest.num_rows else []
        self.pending_rows = rest.num_rows

    def end_cell(self) -> None:
        self._drain(final=True)
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.part = 0

    def add(self, cell: int, table: pa.Table) -> None:
        if cell != self.cell:
            if self.cell is not None:
                self.end_cell()
            self.cell = cell
        self.pending.append(table)
        self.pending_rows += table.num_rows
        if self.pending_rows >= _ROW_GROUP:
            self._drain(final=False)

    def close(self) -> None:
        if self.cell is not None:
            self.end_cell()
        self.cell = None


def _partition_pass(stage: Path, root: Path, schema: pa.Schema, nside: int) -> None:
    """Rewrite staged buckets as ``data/cell=<n>/part-<i>.parquet``, one cell at a time.

    Memory is bounded by the largest stage bucket (a NEST nside-8 pixel, ~54 deg^2):
    a survey concentrated in one such bucket loads that bucket whole.
    """
    shift = 2 * int(np.log2(NSIDE_FINE // nside))
    cw = _CellWriter(root, schema)
    try:
        for bdir in sorted(d for d in stage.iterdir() if d.is_dir()):
            table = _read_bucket(stage, bdir, schema)
            cells = table["hpix32"].to_numpy().astype(np.int64) >> shift
            uniq, start = np.unique(cells, return_index=True)
            stop = np.append(start[1:], len(cells))
            for c, a, z in zip(uniq.tolist(), start.tolist(), stop.tolist(), strict=True):
                cw.add(c, table.slice(a, z - a).select(schema.names))
            del table
        cw.close()
    finally:
        if cw.writer is not None:
            cw.writer.close()


def write_dataset(batches: Iterable[pa.RecordBatch] | pa.Table, out: Path, *,
                  geometry: Geometry, survey: SurveyMeta, loader: LoaderInfo,
                  original_files: Sequence[Path | OriginalFile] = (),
                  partition_nside: int | None = None,
                  extra_stats_columns: Sequence[str] = (),
                  extensions: dict[str, dict] | None = None,
                  overwrite: bool = False, batch_size: int = 250_000) -> Manifest:
    """Write *batches* as an OUF 3.0 dataset at *out* and return its manifest.

    ``out`` existing without ``overwrite`` raises FormatError before anything is
    touched. On overwrite the previous manifest is kept under
    ``_history/<created_utc>.manifest.json`` and ``validity.version`` increments.
    ``hpix32`` is computed from ra/dec when absent. Any failure (including one
    raised by the batch iterator) leaves a pre-existing *out* untouched.

    Memory: the stage pass buffers at most ~128 MiB; the partition pass holds one
    NEST nside-8 bucket (~54 deg^2) at a time, so a survey concentrated in a single
    such bucket is loaded whole. In the ``_index.parquet`` a null ``<col>_min/_max``
    means *unknown* (e.g. all-NaN column): readers must keep such files.
    """
    out = Path(out)
    geometry = Geometry(geometry)
    extensions = dict(extensions or {})
    if partition_nside is not None and not (
            1 <= partition_nside <= NSIDE_FINE and partition_nside & (partition_nside - 1) == 0):
        raise ValidationError(
            f"partition_nside must be a power of 2 in [1, {NSIDE_FINE}], got {partition_nside}")
    previous: Manifest | None = None
    if out.exists():
        if not overwrite:
            raise FormatError(f"{out} already exists; pass overwrite=True to replace it")
        if not (out / MANIFEST_FILENAME).is_file():
            raise FormatError(
                f"{out} exists but has no {MANIFEST_FILENAME}: refusing to overwrite "
                "a directory that is not an OUF dataset")
        try:
            previous = read_manifest(out / MANIFEST_FILENAME)
        except FormatError as e:
            raise FormatError(
                f"{out} is not an OUF 3.0 dataset ({e}); it will not be overwritten: "
                "delete it manually before re-converting") from e
    files = tuple(
        f if isinstance(f, OriginalFile) else OriginalFile(Path(f).name, hash_file(f), None)
        for f in original_files)
    created = _utc_now()
    version = 1 if previous is None else previous.validity.version + 1
    template = Manifest(
        format_version=FORMAT_VERSION, geometry=geometry, survey=survey, loader=loader,
        created_utc=created, schema=pa.schema([]), partitioning=Partitioning(1),
        original_files=files, n_rows=0, validity=Validity(created, None, version),
        extensions=extensions)  # validates the extensions before any I/O

    with atomic_dir(out) as tmp:
        stage = tmp / _STAGE
        stage.mkdir()
        counter = _Counter()
        stream = _validated_stream(batches, geometry, batch_size, counter)
        first = next(stream, None)
        if first is None:
            raise ValidationError("no rows: refusing to write an empty dataset")
        schema = counter.schema
        stats_cols = list(extra_stats_columns)
        for c in stats_cols:
            if c in _STATS_BASE:
                raise ValidationError(
                    f"extra_stats_columns: {c!r} is already a core index column")
            if c not in schema.names:
                raise ValidationError(f"extra_stats_columns: no column {c!r}")
            if not (pa.types.is_integer(schema.field(c).type)
                    or pa.types.is_floating(schema.field(c).type)):
                raise ValidationError(f"extra_stats_columns: {c!r} is not numeric")

        def chain() -> Iterator[pa.RecordBatch]:
            yield first
            yield from stream

        try:
            _stage_stream(chain(), stage)
            n_rows = counter.n_rows
            nside = partition_nside or auto_partition_nside(n_rows)
            _partition_pass(stage, tmp, schema, nside)
        except (pa.ArrowException, OSError) as e:
            raise FormatError(f"writing {out} failed: {e}") from e
        shutil.rmtree(stage)

        index = _build_index(tmp, stats_cols)
        if sum(index["n_rows"].to_pylist()) != n_rows:
            raise FormatError("row count mismatch between staged and partitioned data")

        if previous is not None:
            hist = tmp / _HISTORY
            if (out / _HISTORY).is_dir():
                shutil.copytree(out / _HISTORY, hist)
            else:
                hist.mkdir()
            shutil.copyfile(out / MANIFEST_FILENAME, hist / _hist_name(previous.created_utc))

        manifest = Manifest(
            format_version=FORMAT_VERSION, geometry=geometry, survey=survey, loader=loader,
            created_utc=created, schema=schema, partitioning=Partitioning(nside, True),
            original_files=template.original_files, n_rows=n_rows,
            validity=template.validity, extensions=extensions)
        write_partition_index(tmp / INDEX_FILENAME, index)
        write_manifest(tmp / MANIFEST_FILENAME, manifest)
    return manifest
