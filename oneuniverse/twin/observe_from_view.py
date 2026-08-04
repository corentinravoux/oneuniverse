"""The data → twin socket: grid a Pillar-1 catalog of tracer positions into the
`Observation` a ReconstructionEngine consumes.

This is the first real ``oneuniverse.data`` → ``oneuniverse.twin`` edge (the mock
in ``mock_observe.py`` is the synthetic stand-in it replaces). ``twin`` may import
both pillars; ``simulation`` stays Rule-1 clean.

Scope: box positions (columns x/y/z, Mpc/h). Sky→comoving conversion (ra/dec/z +
fiducial cosmology) is the real-survey extension — deliberately not here, so no
cosmology leaks below the twin call site.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from oneuniverse.simulation.pm.deposit import deposit_cic
from oneuniverse.twin.engine import Observation


def _positions(source, cols: Sequence[str]) -> np.ndarray:
    """Extract an (N,3) float array of box positions from a catalog-like source."""
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("ndarray source must be (N,3) positions")
        return arr
    # DatasetView (has .read) / MeasurementSet PointSet (has .catalog) / DataFrame
    if hasattr(source, "read"):
        df = source.read(columns=list(cols))
    elif hasattr(source, "catalog"):
        df = source.catalog
    else:
        df = source  # assume DataFrame-like
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"catalog missing position columns {missing}; "
                       f"pass position_cols= to match your data")
    return np.column_stack([np.asarray(df[c], float) for c in cols])


def observe_from_view(source, *, box_size: float, n_grid: int,
                      bias: float = 1.0, nbar: Optional[float] = None,
                      position_cols: Sequence[str] = ("x", "y", "z"),
                      mask: Optional[np.ndarray] = None) -> Observation:
    """Grid catalogued tracer positions into an :class:`Observation`.

    Parameters
    ----------
    source : DatasetView | MeasurementSet PointSet | DataFrame | (N,3) ndarray.
    box_size, n_grid : the target mesh (Mpc/h, cells per side).
    bias : linear tracer bias carried into the Observation.
    nbar : mean number density; default = N / box^3.
    position_cols : catalog columns holding box x/y/z.
    mask : optional (n,n,n) selection in [0,1].
    """
    pos = _positions(source, position_cols)
    pos = np.mod(pos, box_size)  # wrap into the periodic box
    counts = deposit_cic(pos, n_grid, box_size)  # mass (≈counts) per cell
    mean = float(counts.mean())
    delta_g = counts / mean - 1.0 if mean > 0 else np.zeros_like(counts)
    if mask is not None:
        delta_g = delta_g * np.asarray(mask, float)
    if nbar is None:
        nbar = len(pos) / box_size ** 3
    return Observation(delta_g=delta_g, nbar=float(nbar), bias=float(bias),
                       mask=mask)
