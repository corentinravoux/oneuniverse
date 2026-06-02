"""Gaussian random density field delta(x) on a regular mesh.

This is the OUF-Sim "field" (mesh / voxel) product. Method: white noise
in real space -> rFFT -> colour by sqrt(P(k)) -> irFFT, with the
Pylians-style normalisation factor 1/sqrt(V_cell) so the discrete
real-space variance approximates (1/V) sum_k P(k). Seeded -> reproducible.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.power_spectrum import linear_power


def generate_density_field(
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    z: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a real (n_grid, n_grid, n_grid) linear density contrast.

    Parameters
    ----------
    box_size
        Comoving box side in Mpc/h.
    n_grid
        Cells per side.
    z
        Redshift (sets the growth-scaled amplitude).
    seed
        RNG seed (reproducible).
    """
    c = require_cosmo(cosmo)
    n = int(n_grid)
    rng = np.random.default_rng(seed)

    # |k| grid (h/Mpc), rfft layout.
    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)

    pk = np.zeros_like(kmag)
    nz = kmag > 0.0
    pk[nz] = linear_power(kmag[nz], c, z=z)

    # White noise in real space, colour in Fourier space.
    white = rng.standard_normal((n, n, n))
    white_k = np.fft.rfftn(white)
    delta_k = white_k * np.sqrt(pk)
    delta = np.fft.irfftn(delta_k, s=(n, n, n))

    # Normalise so variance ~ (1/V) sum_k P(k): factor 1/sqrt(V_cell).
    v_cell = (box_size / n) ** 3
    delta *= 1.0 / np.sqrt(v_cell)
    # Enforce zero mean (DC mode).
    delta -= delta.mean()
    return delta
