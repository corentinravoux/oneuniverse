"""Wiener-filter reconstruction of the matter field from a tracer field.

δ̂_m(k) = [b·P_m(k) / (b²·P_m(k) + N)] · δ_g(k), N = 1/n̄ shot noise.
Full-box (periodic) → diagonal in Fourier space. Masked / non-periodic
reconstruction (messy, mode-coupling) is a later complexification.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.power_spectrum import linear_power


def _kgrid(n, box):
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)


def wiener_reconstruct(delta_g, cosmo: CosmologySpec, *, box_size, nbar,
                       bias=1.0, z=0.0) -> np.ndarray:
    """Minimum-variance estimate of the matter field from ``delta_g``."""
    d = np.asarray(delta_g, dtype=np.float64)
    n = d.shape[0]
    kmag = _kgrid(n, box_size)
    Pm = np.zeros_like(kmag)
    nz = kmag > 0
    Pm[nz] = linear_power(kmag[nz], cosmo, z=z)
    N = 1.0 / nbar
    denom = bias * bias * Pm + N
    gain = np.zeros_like(Pm)
    good = denom > 0
    gain[good] = (bias * Pm[good]) / denom[good]
    dk = np.fft.rfftn(d)
    return np.fft.irfftn(gain * dk, s=(n, n, n))
