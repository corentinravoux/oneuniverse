"""Toy GR / peculiar-potential field: solve ∇²φ = δ in Fourier space.

The OUF-Sim ``gr_fields`` product — a stand-in for stored metric / potential
fields. φ_k = −δ_k / k² (k=0 → 0). Stored as a regular-grid field like the
density (memmap tiles). Also the long-range-force provider reused by the
resimulation far-field (S8).
"""
from __future__ import annotations

import numpy as np


def potential_field(delta, *, box_size) -> np.ndarray:
    """Peculiar potential φ with ∇²φ = δ (zero-mean, periodic)."""
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    k2[0, 0, 0] = 1.0
    phik = -np.fft.rfftn(d) / k2
    phik[0, 0, 0] = 0.0
    return np.fft.irfftn(phik, s=(n, n, n))


def laplacian(field, *, box_size) -> np.ndarray:
    """∇²field via FFT (for verification)."""
    f = np.asarray(field, dtype=np.float64)
    n = f.shape[0]
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    return np.fft.irfftn(-k2 * np.fft.rfftn(f), s=(n, n, n))
