"""Mock survey geometry + selection (the first realistic 'data' shape).

Still fully synthetic — no ``oneuniverse.data`` import yet (that is the next
data-side complexification). Provides selection fields in [0,1] (a hard
footprint mask, or a smoothly-declining radial completeness ~ a flux-limited
n(z)) that ``mock_tracer_field(..., mask=)`` consumes.

The diagonal full-box Wiener is only an approximation under a mask (the mask
couples Fourier modes); a rigorous masked solve (conjugate-gradient /
messenger field) is a later complexification. Inside the footprint, away
from the edges, large-scale recovery survives.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def slab_mask(n_grid: int, *, frac: float = 0.5, axis: int = 2) -> np.ndarray:
    """A slab footprint: cells with index < frac·n along ``axis`` are in."""
    m = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
    w = max(1, int(round(frac * n_grid)))
    sl = [slice(None)] * 3
    sl[axis] = slice(0, w)
    m[tuple(sl)] = 1.0
    return m


def _radius_grid(n_grid, box_size, center):
    c = (np.asarray(center, float) if center is not None
         else np.full(3, box_size / 2.0))
    g = (np.arange(n_grid) + 0.5) * (box_size / n_grid)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    return np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2)


def ball_mask(n_grid: int, *, box_size: float, radius: float,
              center: Optional[Sequence[float]] = None) -> np.ndarray:
    """A spherical survey volume of ``radius`` about ``center`` (box centre)."""
    r = _radius_grid(n_grid, box_size, center)
    return (r <= radius).astype(np.float64)


def radial_completeness(n_grid: int, *, box_size: float, r_scale: float,
                        center: Optional[Sequence[float]] = None) -> np.ndarray:
    """Mock n(z): completeness exp(−r/r_scale), 1 at the observer, in [0,1]."""
    r = _radius_grid(n_grid, box_size, center)
    return np.exp(-r / r_scale)
