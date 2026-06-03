"""Zoom initial conditions — refine a coarse IC to higher resolution.

The MUSIC/Panphasia idea: keep the parent's large-scale modes (k < parent
Nyquist) and **add new small-scale power** (k up to the fine Nyquist) drawn
from P(k) with new phases. This is what gives a resimulation real *fidelity
gain* (more resolved small-scale structure), as opposed to a same-resolution
re-run.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field


def refine_ic(coarse: np.ndarray, *, box_sub: float, cosmo: CosmologySpec,
              factor: int, seed: int = 0) -> np.ndarray:
    """Return a (factor·n)³ IC: coarse low-k modes + new fine small-scale power."""
    n_c = coarse.shape[0]
    n_f = factor * n_c
    # a fine field carrying the full P(k) up to the fine Nyquist
    fine = generate_density_field(cosmo, box_size=box_sub, n_grid=n_f,
                                  z=0.0, seed=seed)
    ck = np.fft.fftshift(np.fft.fftn(coarse))
    fk = np.fft.fftshift(np.fft.fftn(fine))
    # overwrite the fine grid's low-k modes (centre block) with the coarse
    # modes (the parent's constrained large scales); FFT amplitude ∝ N³.
    c0 = (n_f - n_c) // 2
    fk[c0:c0 + n_c, c0:c0 + n_c, c0:c0 + n_c] = ck * float(factor) ** 3
    return np.fft.ifftn(np.fft.ifftshift(fk)).real
