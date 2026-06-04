"""FFT Poisson force on the mesh.

Given the overdensity delta, the peculiar potential solves ∇²φ = δ
(φ_k = −δ_k/k²); the force is g = −∇φ → g_k = i k δ_k / k². Returns the
three force-component grids (zero-mean, periodic). Amplitude is in the
toy convention where δ sources the force directly; the integrator carries
the cosmological prefactors.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def pm_force_isolated(delta: np.ndarray, box: float) -> Tuple[np.ndarray, ...]:
    """Open (non-periodic) force via the Hockney zero-padded FFT convolution.

    Embeds the n³ density in a 2n³ zero-padded grid and convolves with the
    real-space force kernel g_i(r) = −x_i/(4π r³), then crops the central n³.
    This removes the periodic images that a plain FFT-Poisson injects — the
    Dirichlet-style boundary sCOLA tiles need.
    """
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    N = 2 * n
    cell = box / n
    dpad = np.zeros((N, N, N)); dpad[:n, :n, :n] = d
    p = np.fft.fftfreq(N, d=1.0 / N) * cell            # positions, kernel @ 0
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2); r[0, 0, 0] = 1.0
    inv = 1.0 / (4.0 * np.pi * r ** 3)
    dpk = np.fft.fftn(dpad)
    out = []
    for comp in (X, Y, Z):
        g = -comp * inv; g[0, 0, 0] = 0.0              # zero self-force
        f = np.real(np.fft.ifftn(dpk * np.fft.fftn(g))) * cell ** 3
        out.append(f[:n, :n, :n])
    return tuple(out)


def pm_force(delta: np.ndarray, box: float) -> Tuple[np.ndarray, ...]:
    n = delta.shape[0]
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    k2[0, 0, 0] = 1.0
    dk = np.fft.rfftn(delta)
    forces = []
    for kg in (kxg, kyg, kzg):
        gk = 1j * kg / k2 * dk
        gk[0, 0, 0] = 0.0
        forces.append(np.fft.irfftn(gk, s=(n, n, n)))
    return tuple(forces)
