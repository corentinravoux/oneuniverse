"""Fourier-space verification: cross-correlation r(k) and power ratio.

S9 consolidation: the mode binning delegates to the canonical core
:func:`oneuniverse.simulation.validation.binned_mode_powers`; this module
keeps only the twin k-convention (fundamental-mode-spaced edges
``kf/2 + i·kf``) and the NaN-for-empty-bins presentation. Numerics are
identical to the pre-consolidation implementation (same edges, same digitize
semantics, sums-ratios == means-ratios).
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.validation import binned_mode_powers


def _bin_kgrid(n, box):
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)


def _bins(kmag, box, n):
    kf = 2.0 * np.pi / box
    kny = np.pi * n / box
    edges = np.arange(kf / 2, kny, kf)
    idx = np.digitize(kmag.ravel(), edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return idx, edges, centres


def _kf_edges(n, box):
    kf = 2.0 * np.pi / box
    kny = np.pi * n / box
    return np.arange(kf / 2, kny, kf)


def cross_correlation(a, b, *, box_size):
    """Binned cross-correlation coefficient r(k) of two real fields."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    edges = _kf_edges(a.shape[0], box_size)
    centres, S_aa, S_bb, S_ab, _ = binned_mode_powers(
        a, b, box=box_size, edges=edges)
    r = np.full(len(centres), np.nan)
    denom = np.sqrt(S_aa * S_bb)
    good = denom > 0
    r[good] = S_ab[good] / denom[good]
    return centres, r


def power_ratio(a, b, *, box_size):
    """Binned P_a(k)/P_b(k)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    edges = _kf_edges(a.shape[0], box_size)
    centres, S_aa, S_bb, _, _ = binned_mode_powers(
        a, b, box=box_size, edges=edges)
    ratio = np.full(len(centres), np.nan)
    good = S_bb > 0
    ratio[good] = S_aa[good] / S_bb[good]
    return centres, ratio
