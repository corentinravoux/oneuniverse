"""Resimulation verification gates.

Gate 1 (pre-run, necessary): the mini-sim IC's large-scale density matches
the parent on the shared sub-volume. Automatic if the extraction is phase-
consistent — a unit test on the linkage, not a physics result. Fails for a
phase-scrambled IC.

(Gate 2/3 — post-run sufficient + error budget — land in S8.5.)
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _cross_r(a, b, box_size, n_bins=12):
    """Binned cross-correlation r(k) of two equal-shape real fields."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kx = np.fft.fftfreq(n, d=box_size / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2).ravel()
    edges = np.linspace(kmag[kmag > 0].min(), kmag.max(), n_bins + 1)
    idx = np.digitize(kmag, edges)
    cross = np.real(ak * np.conj(bk)).ravel()
    pa = (np.abs(ak) ** 2).ravel(); pb = (np.abs(bk) ** 2).ravel()
    centres, r = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() and pa[m].sum() > 0 and pb[m].sum() > 0:
            centres.append(0.5 * (edges[i - 1] + edges[i]))
            r.append(cross[m].sum() / np.sqrt(pa[m].sum() * pb[m].sum()))
    return np.array(centres), np.array(r)


def gate1_consistency(mini_ic: np.ndarray, parent_sub: np.ndarray, *,
                      box_size: float) -> Dict:
    """Pre-run check that the mini IC matches the parent on the sub-volume."""
    k, r = _cross_r(mini_ic, parent_sub, box_size)
    cell_corr = float(np.corrcoef(mini_ic.ravel(), parent_sub.ravel())[0, 1])
    low = r[k < np.median(k)]
    passed = bool(cell_corr > 0.95 and np.nanmedian(low) > 0.9)
    return {"k": k, "r": r, "cell_corr": cell_corr,
            "r_lowk": float(np.nanmedian(low)), "passed": passed}
