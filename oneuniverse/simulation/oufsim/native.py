"""Layer-2 native-format readers for the wrap-in-place (`reference`) projection.

A `reference` store holds only manifest + sidecar index; the bulk data stays
in the native files and is read through a `NativeReaderAdapter` (memmap /
partial read). The dummy ships two formats — `npy` (scattered linear layout)
and `packed_npy` (chunk-sorted slab) — registered in `ADAPTERS`. A real
backend (parallel-HDF5, ASDF/pack9, BigFile) implements the same ABC and
registers itself; that is how a petabyte sim is wrapped without copying.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np

CellSlice = Tuple[slice, slice, slice]


class NativeReaderAdapter(abc.ABC):
    """Partial reader over a native simulation file (no whole-array load)."""

    native_format: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def read_field_region(self, path: Union[str, Path],
                          cell_slice: CellSlice) -> np.ndarray:
        """Return a sub-array of a native 3-D field (memmap-backed)."""

    def read_rows(self, path: Union[str, Path], row_slice: slice,
                  columns: Optional[Sequence[str]] = None
                  ) -> Dict[str, np.ndarray]:
        """Return {column: array} for a contiguous row range of a point product.

        Optional capability: formats without a row product (e.g. a bare field
        `.npy`) leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no row product (read_rows)")


ADAPTERS: Dict[str, NativeReaderAdapter] = {}


def register_adapter(cls):
    """Class decorator: register an adapter instance by its ``native_format``."""
    fmt = getattr(cls, "native_format", None)
    if not fmt or fmt == "abstract":
        raise ValueError(f"{cls.__name__} must set a concrete native_format")
    ADAPTERS[fmt] = cls()
    return cls


def get_adapter(native_format: str) -> NativeReaderAdapter:
    if native_format not in ADAPTERS:
        raise KeyError(
            f"no native adapter for format {native_format!r}; "
            f"known: {sorted(ADAPTERS)}")
    return ADAPTERS[native_format]


@register_adapter
class NumpyFieldAdapter(NativeReaderAdapter):
    """numpy `.npy` field adapter — the scattered linear native format."""

    native_format = "npy"

    def read_field_region(self, path, cell_slice):
        arr = np.load(path, mmap_mode="r")
        return np.array(arr[cell_slice])     # materialise only the sub-region
