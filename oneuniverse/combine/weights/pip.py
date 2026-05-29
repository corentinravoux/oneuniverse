"""
oneuniverse.combine.weights.pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pairwise-inverse-probability (PIP) bitweight expansion for DESI
clustering.

DESI ships ``BITWEIGHTS: i8[N]`` per object: bit ``k`` is 1 iff the
object passed fiber assignment in realisation ``k``. Two output modes:

* ``"fraction"`` (default): per-row fractional weight
  ``count_set_bits / (64 * N)`` — a scalar between 0 and 1 suitable
  as a drop-in object weight.
* ``"realisations"``: per-row ``(64 * N,)`` array of 0/1 floats, one
  per PIP realisation, for jackknife-style accumulators.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight

_ALLOWED_MODES = frozenset({"fraction", "realisations"})


class PipBitweightWeight(Weight):
    """PIP bitweight expansion of ``i8[N]`` arrays.

    Parameters
    ----------
    bitweights_col : str
        Column carrying the per-row ``i8[N]`` BITWEIGHTS payload.
        Default ``"BITWEIGHTS"``.
    mode : str
        ``"fraction"`` (default) or ``"realisations"``.
    name : str or None
        Override for ``repr``.
    """

    def __init__(
        self,
        bitweights_col: str = "BITWEIGHTS",
        mode: str = "fraction",
        name: Optional[str] = None,
    ) -> None:
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"unknown PipBitweightWeight mode {mode!r}; "
                f"allowed: {sorted(_ALLOWED_MODES)}"
            )
        self.bitweights_col = bitweights_col
        self.mode = mode
        self.name = name or f"pip({mode})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.bitweights_col not in df.columns:
            raise KeyError(
                f"PipBitweightWeight: missing column "
                f"{self.bitweights_col!r}"
            )
        rows = df[self.bitweights_col].to_numpy()
        n_rows = len(rows)
        first = np.asarray(rows[0], dtype="i8")
        n_ints = first.size
        n_bits = 64 * n_ints
        stacked = np.empty((n_rows, n_ints), dtype="i8")
        for i, r in enumerate(rows):
            stacked[i, :] = np.asarray(r, dtype="i8").reshape(-1)
        # unpackbits expects uint8 and unpacks each byte to 8 bits.
        bits = np.unpackbits(
            stacked.view(np.uint8).reshape(n_rows, -1),
            axis=1,
        ).astype(np.float64)
        bits = bits.reshape(n_rows, n_bits)
        if self.mode == "fraction":
            return bits.sum(axis=1) / float(n_bits)
        return bits
