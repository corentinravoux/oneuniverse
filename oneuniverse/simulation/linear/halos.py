"""Toy halo catalogue from local maxima of the density field.

This is the OUF-Sim "halos" product. A "halo" is a cell whose delta
exceeds ``threshold`` AND is a strict local maximum over its 26
neighbours (periodic). Mass is a toy proxy: (1 + delta) * mean cell
mass, with mean cell mass set from rho_crit * V_cell. The result is a
plain dict of equal-length arrays (parquet-friendly).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# rho_crit = 2.775e11 h^2 Msun / Mpc^3 -> in Msun/h per (Mpc/h)^3 the
# h-factors cancel, leaving this numeric constant.
_RHO_CRIT_H2 = 2.775e11  # Msun/h / (Mpc/h)^3


def find_peaks(
    delta: np.ndarray,
    *,
    box_size: float,
    threshold: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Return a toy halo catalogue as a dict of arrays."""
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    cell = box_size / n

    # Local-maximum test over 26 periodic neighbours.
    is_peak = np.ones_like(d, dtype=bool)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                shifted = np.roll(np.roll(np.roll(d, sx, 0), sy, 1), sz, 2)
                is_peak &= d > shifted

    mask = is_peak & (d > threshold)
    idx = np.argwhere(mask)  # (n_halos, 3) integer cell indices
    if idx.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "halo_id": np.empty(0, dtype=np.int64),
            "x": empty_f, "y": empty_f, "z": empty_f,
            "delta_peak": empty_f, "mass": empty_f,
        }

    centres = (idx + 0.5) * cell
    deltas = d[mask]
    mean_cell_mass = _RHO_CRIT_H2 * cell ** 3  # toy proxy
    mass = (1.0 + deltas) * mean_cell_mass
    return {
        "halo_id": np.arange(idx.shape[0], dtype=np.int64),
        "x": centres[:, 0].astype(np.float64),
        "y": centres[:, 1].astype(np.float64),
        "z": centres[:, 2].astype(np.float64),
        "delta_peak": deltas.astype(np.float64),
        "mass": mass.astype(np.float64),
    }
