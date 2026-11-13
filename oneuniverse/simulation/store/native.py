"""Adapters reading native simulation files by row range or grid region.

An adapter hides the on-disk layout of a simulation's own files so the store
writers (re-encode or index-only "reference" projection) and, later, readers can
address them uniformly.  Two adapters ship:

``npy``
    One structured ``.npy`` array per catalog (columns are the dtype fields,
    native dtypes kept) or one 3-D ``.npy`` grid per field.  Memory-mapped.
``packed_npy``
    ``npy`` particle slabs *sorted by chunk id*, with a ``header.json`` next to
    them: ``{"box_size", "n_side", "redshifts", "chunks": {ztag: [[chunk_id,
    start, stop], ...]}}``.  The header makes an index-only store possible.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.registry import Registry


class NativeReaderAdapter(ABC):
    """Reads rows of a catalog file or sub-regions of a field file."""

    format: ClassVar[str]

    @abstractmethod
    def num_rows(self, path: str | Path) -> int:
        """Number of rows of the catalog at ``path``."""

    @abstractmethod
    def read_rows(self, path: str | Path, start: int, stop: int, *,
                  columns: Sequence[str] | None = None) -> pa.Table:
        """Rows ``[start, stop)`` of the catalog at ``path``."""

    @abstractmethod
    def field_shape(self, path: str | Path) -> tuple[int, int, int]:
        """Shape of the grid at ``path``."""

    @abstractmethod
    def read_field_region(self, path: str | Path, lo: tuple[int, int, int],
                          hi: tuple[int, int, int]) -> np.ndarray:
        """Cells ``[lo, hi)`` (half-open index ranges) of the grid at ``path``."""


def _memmap(path: str | Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise FormatError(f"cannot open native array {path}: {exc}") from exc


class NpyFieldAdapter(NativeReaderAdapter):
    """Plain ``.npy`` files: structured arrays as catalogs, 3-D arrays as grids."""

    format = "npy"

    def _table_array(self, path) -> np.ndarray:
        arr = _memmap(path)
        if arr.dtype.names is None or arr.ndim != 1:
            raise FormatError(f"{path} is not a 1-D structured array; cannot read rows")
        return arr

    def num_rows(self, path) -> int:
        return int(self._table_array(path).shape[0])

    def read_rows(self, path, start, stop, *, columns=None) -> pa.Table:
        arr = self._table_array(path)
        n = arr.shape[0]
        if not 0 <= start <= stop <= n:
            raise ValidationError(f"row range [{start}, {stop}) outside [0, {n}]")
        names = list(columns) if columns is not None else list(arr.dtype.names)
        missing = [c for c in names if c not in arr.dtype.names]
        if missing:
            raise ValidationError(f"{path} has no column(s) {missing}; has {list(arr.dtype.names)}")
        return pa.table({c: pa.array(np.ascontiguousarray(arr[c][start:stop])) for c in names})

    def field_shape(self, path) -> tuple[int, int, int]:
        shape = _memmap(path).shape
        if len(shape) != 3:
            raise FormatError(f"{path} is not a 3-D grid (shape {shape})")
        return (int(shape[0]), int(shape[1]), int(shape[2]))

    def read_field_region(self, path, lo, hi) -> np.ndarray:
        shape = self.field_shape(path)
        if not all(0 <= a <= b <= n for a, b, n in zip(lo, hi, shape)):
            raise ValidationError(f"region {tuple(lo)}..{tuple(hi)} outside grid {shape}")
        arr = _memmap(path)
        return np.ascontiguousarray(
            arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])


class PackedNpyAdapter(NpyFieldAdapter):
    """Chunk-sorted structured ``.npy`` slab plus ``header.json`` chunk ranges."""

    format = "packed_npy"

    def header(self, path: str | Path) -> dict:
        hdr = Path(path).parent / "header.json"
        try:
            return json.loads(hdr.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FormatError(f"cannot read packed-npy header {hdr}: {exc}") from exc

    def chunk_ranges(self, path: str | Path, z: float) -> dict[int, tuple[int, int]]:
        """``{chunk_id: (row_start, row_stop)}`` of the slab for redshift ``z``."""
        hdr = self.header(path)
        tag = f"{float(z):.3f}"
        try:
            rows = hdr["chunks"][tag]
        except KeyError:
            raise FormatError(f"header of {path} has no chunk ranges for z={tag}") from None
        return {int(c): (int(a), int(b)) for c, a, b in rows}


_ADAPTERS: Registry[type[NativeReaderAdapter]] = Registry(
    "native reader adapter", key=lambda cls: cls.format)


def register_adapter(cls: type[NativeReaderAdapter]) -> type[NativeReaderAdapter]:
    """Register an adapter class under its ``format``; usable as a decorator."""
    return _ADAPTERS.register(cls)


def get_adapter(format: str) -> NativeReaderAdapter:
    """Instance of the adapter registered for ``format`` (``NotFoundError`` if unknown)."""
    return _ADAPTERS.get(format)()


register_adapter(NpyFieldAdapter)
register_adapter(PackedNpyAdapter)
