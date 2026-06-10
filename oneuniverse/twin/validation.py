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

from oneuniverse.simulation.validation import binned_mode_powers
from oneuniverse.twin.verify import (
    _kf_edges,
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
    # S9: delegates to the canonical mode-binning core; same kf edges,
    # T(k) = S_ab/S_bb is identical to the pre-consolidation sums.
    edges = _kf_edges(rec.shape[0], box_size)
    _, _, S_bb, S_ab, _ = binned_mode_powers(rec, truth, box=box_size,
                                             edges=edges)
    out = np.full(len(edges) - 1, np.nan)
    good = S_bb > 0
    out[good] = S_ab[good] / S_bb[good]
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
