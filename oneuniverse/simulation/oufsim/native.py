"""Layer-2 native-format readers for the wrap-in-place (`reference`) projection.

A `reference` store holds only manifest + sidecar index; the bulk data stays
in the native files and is read through a `NativeReaderAdapter` (memmap /
partial read). The dummy uses numpy `.npy`; a real backend (HDF5 parallel,
ASDF/pack9, BigFile) implements the same ABC — that is how a petabyte sim is
wrapped without copying.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Tuple, Union

import numpy as np


class NativeReaderAdapter(abc.ABC):
    """Partial reader over a native simulation file (no whole-array load)."""

    @abc.abstractmethod
    def read_field_region(self, path: Union[str, Path],
                          cell_slice: Tuple[slice, slice, slice]) -> np.ndarray:
        """Return a sub-array of a native 3-D field (memmap-backed)."""


class NumpyFieldAdapter(NativeReaderAdapter):
    """numpy `.npy` adapter — the dummy native format."""

    def read_field_region(self, path, cell_slice):
        arr = np.load(path, mmap_mode="r")
        return np.array(arr[cell_slice])     # materialise only the sub-region
