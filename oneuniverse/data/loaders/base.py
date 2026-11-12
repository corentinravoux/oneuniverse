"""Declarative survey loaders: a loader describes a catalogue, the base class streams it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

import numpy as np
import pyarrow as pa
from astropy.io import fits

from oneuniverse._core.errors import NotFoundError
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.data.manifest import SurveyMeta
from oneuniverse.data.schema import Geometry, cast_to_core, stable_object_id

_HPIX_NSIDE = 32


class SurveyLoader(ABC):
    meta: ClassVar[SurveyMeta]
    geometry: ClassVar[Geometry] = Geometry.POINT
    z_type: ClassVar[str] = "spec"
    version: ClassVar[str] = "1"
    extensions: ClassVar[dict[str, dict]] = {}

    @abstractmethod
    def batches(self, source: Path, *, batch_size: int = 250_000, **kwargs) -> Iterator[pa.RecordBatch]:
        """Yield core-complete record batches."""

    def original_files(self, source: Path) -> list[Path]:
        return []


def _column_to_arrow(col: np.ndarray) -> pa.Array:
    col = np.asarray(col)
    if col.dtype.kind in "SU":
        stripped = np.char.rstrip(col)
        if stripped.dtype.kind == "S":
            stripped = np.char.decode(stripped, "ascii", errors="replace")
        return pa.array(stripped.tolist(), pa.string())
    if col.dtype.kind == "b":
        return pa.array(col, pa.bool_())
    if col.dtype.byteorder == ">":
        col = col.astype(col.dtype.newbyteorder("="))
    col = np.ascontiguousarray(col)
    if col.ndim == 2:
        flat = pa.array(col.reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, col.shape[1])
    return pa.array(col)


def read_fits_batches(path: Path, hdu: int, columns: list[str], batch_size: int) -> Iterator[tuple[int, pa.Table]]:
    """Yield ``(start_row, native-named table)`` slices of a FITS table, memory-mapped."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        nrows = len(data)
        for start in range(0, nrows, batch_size):
            chunk = data[start:min(start + batch_size, nrows)]
            yield start, pa.table({c: _column_to_arrow(chunk[c]) for c in columns})


class FitsCatalogLoader(SurveyLoader):
    filename: ClassVar[str]
    hdu: ClassVar[int] = 1
    columns: ClassVar[dict[str, str]]          # native -> standard
    z_err_column: ClassVar[str | None] = None  # native name; None -> z_err = NaN

    def row_filter(self, native: pa.Table, **options) -> pa.BooleanArray | None:
        return None

    def derive(self, table: pa.Table) -> pa.Table:
        return table

    def original_files(self, source: Path) -> list[Path]:
        return [Path(source) / self.filename]

    def batches(self, source: Path, *, batch_size: int = 250_000, z_min: float | None = None,
                z_max: float | None = None, **options) -> Iterator[pa.RecordBatch]:
        path = Path(source) / self.filename
        if not path.exists():  # eager: raised at call time, not at first next()
            raise NotFoundError(f"expected FITS file not found: {path}")
        return self._stream(path, batch_size, z_min, z_max, options)

    def _stream(self, path: Path, batch_size: int, z_min: float | None,
                z_max: float | None, options: dict) -> Iterator[pa.RecordBatch]:
        read = list(self.columns)
        if self.z_err_column is not None and self.z_err_column not in read:
            read.append(self.z_err_column)
        survey = self.meta.name
        for start, native in read_fits_batches(path, self.hdu, read, batch_size):
            mask = self.row_filter(native, **options)
            if mask is None:
                rows = np.arange(native.num_rows, dtype=np.int64)
            else:
                rows = np.flatnonzero(np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)).astype(np.int64)
            if rows.size == 0:
                continue
            native = native.take(pa.array(rows))
            source_row = rows + start
            tbl = pa.table({new: native[old] for old, new in self.columns.items()})
            z_err = (native[self.z_err_column].cast(pa.float32()) if self.z_err_column
                     else pa.array(np.full(len(rows), np.nan, dtype=np.float32)))
            tbl = self.derive(tbl)
            keep = np.ones(len(rows), dtype=bool)
            z = tbl["z"].to_numpy()
            if z_min is not None:
                keep &= z >= z_min
            if z_max is not None:
                keep &= z <= z_max
            if not keep.all():
                idx = pa.array(np.flatnonzero(keep))
                tbl, z_err, source_row = tbl.take(idx), z_err.take(idx), source_row[keep]
            if source_row.size == 0:
                continue
            n = len(source_row)
            ra, dec = tbl["ra"].to_numpy(), tbl["dec"].to_numpy()
            tbl = tbl.append_column("survey", pa.array([survey] * n))
            tbl = tbl.append_column("z_type", pa.array([self.z_type] * n))
            tbl = tbl.append_column("z_err", z_err)
            tbl = tbl.append_column("source_row", pa.array(source_row))
            tbl = tbl.append_column("object_id", pa.array(stable_object_id(survey, source_row)))
            tbl = tbl.append_column("hpix32", pa.array(radec_to_pix(_HPIX_NSIDE, ra, dec)))
            tbl = cast_to_core(tbl, self.geometry)
            yield from tbl.combine_chunks().to_batches()
