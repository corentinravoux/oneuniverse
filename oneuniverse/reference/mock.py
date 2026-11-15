"""Lognormal Poisson tracer mocks, survey footprints and the mock challenge.

Conventions
-----------
Grid, units and Fourier conventions are those of
:mod:`oneuniverse.reference.linear`.

Lognormal tracers
    ``counts ~ Poisson(n̄·cell³·(1 + δ_LN))`` per cell with
    ``1 + δ_LN = exp(b·(δ − δ̄) − σ²/2)``, ``σ²`` the sample variance of
    ``b·δ`` over the grid and ``δ̄`` its mean (0 for a
    :func:`~oneuniverse.reference.linear.gaussian_field`). For a Gaussian
    ``δ`` this makes ``⟨1 + δ_LN⟩ = 1`` and, by Price's theorem, the cross
    power ``⟨δ_LN δ⟩ = b·P`` at every k: the *cross* bias is ``b`` even when
    ``σ`` is large, while the *auto* power is enhanced by the non-linear
    transform. Counts are returned as a float64 grid of non-negative integer
    values; the observed contrast is ``counts/⟨counts⟩ − 1`` (inside the
    footprint for a masked survey).

Footprints
    :func:`mask_slab` is true for cell centres ``(i + 0.5)/n`` with
    ``lo_frac ≤ (i + 0.5)/n < hi_frac`` along ``axis``. :func:`mask_ball` is
    true for cell centres within ``radius_frac`` of ``centre_frac`` by the
    *non-periodic* Euclidean distance in fractional box coordinates (a ball
    near a face is cut, not wrapped). Both return bool ``(n, n, n)`` grids.
"""
from __future__ import annotations

import numbers

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.fieldstats import recover_metrics
from oneuniverse.reference.linear import _check_grid, _cubic, gaussian_field
from oneuniverse.reference.reconstruct import (
    _check_seed,
    _derived_seed,
    _positive,
    wiener_reconstruct,
)
from oneuniverse.simulation.specs import CosmologySpec

__all__ = ["lognormal_tracers", "mask_ball", "mask_slab", "run_mock_challenge"]

_TAG_MOCK_TRACERS = 0x4D4B7472  # ASCII "MKtr"


def lognormal_tracers(delta, *, box: float, nbar: float, bias: float,
                      seed: int) -> np.ndarray:
    """Poisson counts of a lognormal tracer field biased by ``bias`` over ``delta``.

    ``counts ~ Poisson(nbar·cell³·exp(b·(δ − δ̄) − σ²/2))``; see the module
    docstring. Returns float64 counts on the grid of ``delta``.
    """
    d = _cubic(delta, "delta")
    n, box = _check_grid(d.shape[0], box)
    nbar = _positive(nbar, "nbar")
    bias = _positive(bias, "bias")
    seed = _check_seed(seed)
    bd = bias * (d - d.mean())
    intensity = nbar * (box / n) ** 3 * np.exp(bd - 0.5 * float(bd.var()))
    return np.random.default_rng(seed).poisson(intensity).astype(np.float64)


def _check_n(n_grid) -> int:
    if isinstance(n_grid, bool) or not isinstance(n_grid, numbers.Integral) or n_grid < 1:
        raise ValidationError(f"n_grid must be a positive integer, got {n_grid!r}")
    return int(n_grid)


def _centres(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def mask_slab(n_grid: int, axis: int, lo_frac: float, hi_frac: float) -> np.ndarray:
    """Bool grid true where ``lo_frac ≤ (i + 0.5)/n < hi_frac`` along ``axis``."""
    n = _check_n(n_grid)
    if axis not in (0, 1, 2):
        raise ValidationError(f"axis must be 0, 1 or 2, got {axis!r}")
    lo, hi = float(lo_frac), float(hi_frac)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValidationError(f"need 0 <= lo_frac < hi_frac <= 1, got {lo_frac}, {hi_frac}")
    c = _centres(n)
    sel = (c >= lo) & (c < hi)
    shape = [1, 1, 1]
    shape[axis] = n
    return np.broadcast_to(sel.reshape(shape), (n, n, n)).copy()


def mask_ball(n_grid: int, centre_frac, radius_frac: float) -> np.ndarray:
    """Bool grid true for cell centres within ``radius_frac`` of ``centre_frac``.

    Non-periodic Euclidean distance in fractional box coordinates.
    """
    n = _check_n(n_grid)
    centre = np.asarray(centre_frac, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValidationError(f"centre_frac must be 3 finite numbers, got {centre_frac!r}")
    radius = _positive(radius_frac, "radius_frac")
    c = _centres(n)
    r2 = ((c[:, None, None] - centre[0]) ** 2 + (c[None, :, None] - centre[1]) ** 2
          + (c[None, None, :] - centre[2]) ** 2)
    return r2 <= radius ** 2


def run_mock_challenge(cosmo: CosmologySpec, *, box: float, n_grid: int, nbar: float,
                       bias: float = 1.5, z: float = 0.0, seed: int) -> dict:
    """Truth → lognormal tracers → Wiener filter → recovery metrics.

    ``truth = gaussian_field(cosmo, box, n_grid, z, seed=seed)``; tracers are
    sampled from a stream derived from ``seed`` (never the truth's);
    ``delta_g = counts/⟨counts⟩ − 1``; ``wiener`` is
    :func:`wiener_reconstruct` of ``delta_g``. Returns exactly
    ``{"truth", "delta_g", "wiener", "metrics"}`` with ``metrics =
    recover_metrics(wiener, truth, box=box)``.
    """
    seed = _check_seed(seed)
    nbar = _positive(nbar, "nbar")
    bias = _positive(bias, "bias")
    truth = gaussian_field(cosmo, box=box, n_grid=n_grid, z=z, seed=seed)
    counts = lognormal_tracers(truth, box=box, nbar=nbar, bias=bias,
                               seed=_derived_seed(_TAG_MOCK_TRACERS, seed))
    mean = counts.mean()
    if mean <= 0:
        raise ValidationError("no tracer sampled: increase nbar or the box")
    delta_g = counts / mean - 1.0
    wiener = wiener_reconstruct(delta_g, cosmo, box=box, nbar=nbar, bias=bias, z=z)
    return {"truth": truth, "delta_g": delta_g, "wiener": wiener,
            "metrics": recover_metrics(wiener, truth, box=box)}
