"""Mask-aware Wiener filter and seed-explicit Hoffman–Ribak constrained realisation.

Conventions
-----------
Grid, units and Fourier normalisation are those of
:mod:`oneuniverse.reference.linear` (full complex FFT, ``P = |δ_k|²·V/N²``,
cell-centred values, ``k`` in h/Mpc, ``P`` in (Mpc/h)³).

Wiener filter
    Diagonal in Fourier space (periodic box)::

        δ_WF,k = S_k / (S_k + N) · δ_g,k / b,   S_k = b²·P_lin(|k|, z),   N = 1/n̄

    ``P_lin(k, z) = linear_power(k, cosmo, z)`` (``∝ D(z)²``); ``N = 1/n̄`` is
    the Poisson shot noise in the same ``V/N²`` power units. The output is the
    *matter* field estimate (divided by the bias ``b``). The gain depends on
    ``|k|`` only, so Hermitian symmetry is preserved and the result is real;
    the k = 0 mode is set to 0 (the mean density is not constrained).

Mask
    ``mask`` is a boolean footprint (or a 0/1 array) of the grid's shape; a
    cell is observed where it is true. Before filtering, ``δ_g`` is
    renormalised by its mean inside the footprint::

        δ_g ← (1 + δ_g) / ⟨1 + δ_g⟩_mask − 1   on footprint cells,   0 outside

    (values given outside the footprint are ignored). The filter is then the
    *simple* masked Wiener: the zero-filled field goes through the same
    diagonal filter, no mode-coupling correction, no effective-noise boost
    outside the footprint. Outside the footprint the estimate therefore only
    carries the filter kernel's leakage from the footprint edge and relaxes to
    the prior mean 0 away from it. ``mask=None`` is identical to an all-true
    footprint. Deep inside the footprint the estimate matches the unmasked
    filter (the kernel is compact: ``1/b`` minus a high-pass correction).

Constrained realisation (Hoffman & Ribak 1991)
    ``δ_CR = δ_WF(δ_g) + (δ_rand − δ_WF(d_mock))``, ``d_mock = b·δ_rand + ε``.
    ``δ_rand`` is a :func:`~oneuniverse.reference.linear.gaussian_field` of
    ``P_lin(k, z)``; the noise ``ε`` is *Gaussian* white noise with per-cell
    variance ``1/(n̄·cell³)``, i.e. power ``1/n̄``, the Gaussian stand-in for
    Poisson noise assumed by the filter. ``d_mock`` is zero-filled outside the
    footprint and filtered by exactly the same :func:`wiener_reconstruct`
    (renormalisation included) as the data. ``seed`` is keyword-only and has
    no default; both random streams derive from it through
    ``np.random.SeedSequence([tag, seed])`` with fixed per-purpose tags. Hence
    the CR depends only on ``δ_g`` and ``seed``: it never sees the truth's
    seed, and even passing the truth's integer seed does not reproduce the
    truth's white noise (``gaussian_field(seed=s)`` draws from
    ``SeedSequence(s)``, a different stream). The old engine reused the truth
    seed and so "recovered" small scales the data cannot constrain.
"""
from __future__ import annotations

import numbers

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.linear import (
    _check_grid,
    _cubic,
    gaussian_field,
    k_grid,
    linear_power,
)
from oneuniverse.simulation.specs import CosmologySpec

__all__ = ["constrained_realization", "wiener_reconstruct"]

# Per-purpose SeedSequence tags (ASCII "CRsg", "CRns").
_TAG_CR_SIGNAL = 0x43527367
_TAG_CR_NOISE = 0x43526E73


def _positive(value, name: str) -> float:
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValidationError(f"{name} must be a positive finite number, got {value}")
    return x


def _check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValidationError(f"seed must be a non-negative integer, got {seed!r}")
    return int(seed)


def _derived_seed(tag: int, seed: int) -> int:
    """An integer seed of the stream ``SeedSequence([tag, seed])``."""
    return int(np.random.SeedSequence([tag, seed]).generate_state(1, dtype=np.uint64)[0])


def _check_mask(mask, shape: tuple[int, ...]) -> np.ndarray:
    """Footprint as a bool grid of ``shape``; all-true when ``mask`` is None."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(mask)
    if m.shape != shape:
        raise ValidationError(f"mask shape {m.shape} differs from the field shape {shape}")
    if m.dtype != bool:
        if not np.all((m == 0) | (m == 1)):
            raise ValidationError("mask must be boolean or contain only 0 and 1")
        m = m.astype(bool)
    if not m.any():
        raise ValidationError("mask selects no cell")
    return m


def _renormalise(delta_g: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """``(1 + δ)/⟨1 + δ⟩_mask − 1`` on footprint cells, 0 elsewhere."""
    one_plus = 1.0 + delta_g[footprint]
    mean = float(one_plus.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValidationError(
            f"mean of 1 + delta_g inside the footprint must be positive, got {mean}")
    out = np.zeros_like(delta_g)
    out[footprint] = one_plus / mean - 1.0
    return out


def _wiener_gain(cosmo: CosmologySpec, n: int, box: float, nbar: float,
                 bias: float, z: float) -> np.ndarray:
    """``S/(S + N)/b`` on the full-FFT grid, 0 at k = 0."""
    _, _, _, kmag = k_grid(n, box)
    signal = np.zeros_like(kmag)
    nz = kmag > 0.0
    signal[nz] = bias ** 2 * linear_power(kmag[nz], cosmo, z=z)
    return signal / (signal + 1.0 / nbar) / bias


def wiener_reconstruct(delta_g, cosmo: CosmologySpec, *, box: float, nbar: float,
                       bias: float = 1.0, z: float = 0.0,
                       mask: np.ndarray | None = None) -> np.ndarray:
    """Wiener-filter estimate of the matter density contrast from ``delta_g``.

    ``δ_WF,k = S/(S+N)·δ_g,k/b`` with ``S = b²·P_lin(k, z)``, ``N = 1/nbar``,
    after renormalising ``delta_g`` inside the footprint ``mask`` (see the
    module docstring). Returns a float64 ``(n, n, n)`` grid with zero mean.
    """
    d = _cubic(delta_g, "delta_g")
    n, box = _check_grid(d.shape[0], box)
    nbar = _positive(nbar, "nbar")
    bias = _positive(bias, "bias")
    footprint = _check_mask(mask, d.shape)
    d = _renormalise(d, footprint)
    gain = _wiener_gain(cosmo, n, box, nbar, bias, float(z))
    return np.ascontiguousarray(np.fft.ifftn(gain * np.fft.fftn(d)).real)


def constrained_realization(delta_g, cosmo: CosmologySpec, *, box: float, nbar: float,
                            bias: float, z: float, seed: int,
                            mask: np.ndarray | None = None) -> np.ndarray:
    """Hoffman–Ribak constrained realisation of the matter field given ``delta_g``.

    ``δ_CR = δ_WF(δ_g) + (δ_rand − δ_WF(b·δ_rand + ε))`` with ``δ_rand`` a
    Gaussian realisation of ``P_lin(k, z)`` and ``ε`` Gaussian noise of power
    ``1/nbar``, both drawn from streams derived from ``seed`` only (required,
    keyword-only, no default). The mock data are masked and renormalised like
    the data. Large scales follow the data; scales the data cannot constrain
    are a fresh draw with the prior power.
    """
    seed = _check_seed(seed)
    d = _cubic(delta_g, "delta_g")
    n, box = _check_grid(d.shape[0], box)
    nbar = _positive(nbar, "nbar")
    bias = _positive(bias, "bias")
    footprint = _check_mask(mask, d.shape)
    kw = {"box": box, "nbar": nbar, "bias": bias, "z": float(z), "mask": footprint}

    wf_data = wiener_reconstruct(d, cosmo, **kw)
    delta_rand = gaussian_field(cosmo, box=box, n_grid=n, z=float(z),
                                seed=_derived_seed(_TAG_CR_SIGNAL, seed))
    rng = np.random.default_rng(np.random.SeedSequence([_TAG_CR_NOISE, seed]))
    noise = rng.standard_normal((n, n, n)) / np.sqrt(nbar * (box / n) ** 3)
    mock = np.where(footprint, bias * delta_rand + noise, 0.0)
    wf_mock = wiener_reconstruct(mock, cosmo, **kw)
    return np.ascontiguousarray(wf_data + delta_rand - wf_mock)
