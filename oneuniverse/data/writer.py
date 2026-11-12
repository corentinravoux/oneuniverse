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
import pyarrow.dataset as ds
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
_HISTORY = "_history"
_ROW_GROUP = 131_072
_STAGE_ROWS_PER_FILE = 1_000_000
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
            lo = float(st.min) if lo is None else min(lo, float(st.min))
            hi = float(st.max) if hi is None else max(hi, float(st.max))
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
        previous = read_manifest(out / MANIFEST_FILENAME)
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
        counter = _Counter()
        stream = _validated_stream(batches, geometry, batch_size, counter)
        first = next(stream, None)
        if first is None:
            raise ValidationError("no rows: refusing to write an empty dataset")

        def chain() -> Iterator[pa.RecordBatch]:
            yield first
            yield from stream
        reader = pa.RecordBatchReader.from_batches(first.schema, chain())
        ds.write_dataset(
            reader, stage, format="parquet", basename_template="stage-{i}.parquet",
            max_rows_per_group=_ROW_GROUP, max_rows_per_file=_STAGE_ROWS_PER_FILE,
            existing_data_behavior="error")
        schema = counter.schema
        n_rows = counter.n_rows
        stats_cols = list(extra_stats_columns)
        for c in stats_cols:
            if c not in schema.names:
                raise ValidationError(f"extra_stats_columns: no column {c!r}")
            if not (pa.types.is_integer(schema.field(c).type)
                    or pa.types.is_floating(schema.field(c).type)):
                raise ValidationError(f"extra_stats_columns: {c!r} is not numeric")

        nside = partition_nside or auto_partition_nside(n_rows)
        shift = 2 * int(np.log2(NSIDE_FINE // nside))
        cell = pc.shift_right(ds.field("hpix32"), shift).cast(pa.int32())
        projection = {n: ds.field(n) for n in schema.names}
        projection["cell"] = cell
        scanner = ds.dataset(stage, format="parquet").scanner(
            columns=projection, batch_size=_ROW_GROUP)
        ds.write_dataset(
            scanner, tmp / DATA_DIR, format="parquet",
            partitioning=ds.partitioning(pa.schema([("cell", pa.int32())]), flavor="hive"),
            basename_template="part-{i}.parquet", max_rows_per_file=MAX_ROWS_PER_FILE,
            max_rows_per_group=_ROW_GROUP, existing_data_behavior="error",
            file_options=ds.ParquetFileFormat().make_write_options(
                compression="zstd", write_statistics=True, use_dictionary=True))
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
