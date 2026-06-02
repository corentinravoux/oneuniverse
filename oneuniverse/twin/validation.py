"""Standard recovery metrics for the twin loop.

One harness every coupling increment (mask, RSD, constrained realisation,
PM, SBI) reports through, so progress is *measurable* and regression-tested
— the methods-paper spine. Built on the C1 Fourier estimators.

- r(k)     : cross-correlation coefficient reconstruction × truth.
- transfer : T(k) = ⟨rec·truth⟩ / ⟨truth²⟩  (amplitude recovery).
- power_ratio : P_rec(k) / P_truth(k).
- k_half   : smallest k where r(k) < 0.5 (the reconstruction scale; nan if
  the field is recovered to the Nyquist frequency).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oneuniverse.twin.verify import (
    _bin_kgrid,
    _bins,
    cross_correlation,
    power_ratio,
)


@dataclass(frozen=True)
class RecoveryMetrics:
    k: np.ndarray
    r: np.ndarray
    transfer: np.ndarray
    power_ratio: np.ndarray
    k_half: float


def _transfer(rec, truth, box_size):
    n = rec.shape[0]
    rk = np.fft.rfftn(rec); tk = np.fft.rfftn(truth)
    kmag = _bin_kgrid(n, box_size)
    idx, edges, centres = _bins(kmag, box_size, n)
    cross = np.real(rk * np.conj(tk)).ravel()
    pt = (np.abs(tk) ** 2).ravel()
    out = np.full(len(centres), np.nan)
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() and pt[m].sum() > 0:
            out[i - 1] = cross[m].sum() / pt[m].sum()
    return out


def _k_half(k, r):
    good = np.isfinite(r)
    k, r = k[good], r[good]
    below = np.where(r < 0.5)[0]
    return float(k[below[0]]) if len(below) else float("nan")


def recover_metrics(rec, truth, *, box_size) -> RecoveryMetrics:
    rec = np.asarray(rec, float); truth = np.asarray(truth, float)
    k, r = cross_correlation(rec, truth, box_size=box_size)
    _, ratio = power_ratio(rec, truth, box_size=box_size)
    transfer = _transfer(rec, truth, box_size)
    return RecoveryMetrics(k=k, r=r, transfer=transfer, power_ratio=ratio,
                           k_half=_k_half(k, r))
