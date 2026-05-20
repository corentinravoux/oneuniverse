"""Synthetic HEALPix maps used by Phase 11 weight tests."""
from __future__ import annotations

import healpy as hp
import numpy as np


def make_footprint_mask(
    nside: int = 32, fsky: float = 0.2, seed: int = 0,
) -> np.ndarray:
    """Binary footprint: every pixel in a dec band gets 1, else 0."""
    npix = hp.nside2npix(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    dec = 90.0 - np.degrees(theta)
    band = (dec > -20) & (dec < 40)
    m = np.zeros(npix, dtype=np.float64)
    m[band] = 1.0
    return m


def make_smooth_completeness(nside: int = 32, seed: int = 0) -> np.ndarray:
    """Per-pixel completeness in [0, 1]: mean 0.9, Gaussian dimples."""
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    base = np.full(npix, 0.9)
    ripple = 0.1 * rng.standard_normal(npix)
    return np.clip(base + ripple, 0.0, 1.0)


def make_systematic_map(nside: int = 32, seed: int = 0) -> np.ndarray:
    """SYSNet-like positive weight map (mean ~ 1.0, log-Gaussian)."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    return np.exp(0.05 * rng.standard_normal(npix))
