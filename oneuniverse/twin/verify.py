"""Fourier-space verification: cross-correlation r(k) and power ratio."""
from __future__ import annotations

import numpy as np


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


def cross_correlation(a, b, *, box_size):
    """Binned cross-correlation coefficient r(k) of two real fields."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kmag = _bin_kgrid(n, box_size)
    idx, edges, centres = _bins(kmag, box_size, n)
    cross = np.real(ak * np.conj(bk)).ravel()
    pa = (np.abs(ak) ** 2).ravel(); pb = (np.abs(bk) ** 2).ravel()
    r = np.full(len(centres), np.nan)
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() == 0:
            continue
        denom = np.sqrt(pa[m].sum() * pb[m].sum())
        if denom > 0:
            r[i - 1] = cross[m].sum() / denom
    return centres, r


def power_ratio(a, b, *, box_size):
    """Binned P_a(k)/P_b(k)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kmag = _bin_kgrid(n, box_size)
    idx, edges, centres = _bins(kmag, box_size, n)
    pa = (np.abs(ak) ** 2).ravel(); pb = (np.abs(bk) ** 2).ravel()
    ratio = np.full(len(centres), np.nan)
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() and pb[m].sum() > 0:
            ratio[i - 1] = pa[m].sum() / pb[m].sum()
    return centres, ratio
