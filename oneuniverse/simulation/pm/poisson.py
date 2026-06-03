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
