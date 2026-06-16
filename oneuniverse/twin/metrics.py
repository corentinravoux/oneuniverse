"""Twin-side field-comparison metrics — the single home (S10 consolidation).

Merges the former ``twin.verify`` (Fourier r(k) / power ratio) and
``twin.validation`` (recovery harness) into one module. ``twin.verify`` and
``twin.validation`` remain as thin compat re-exports so existing imports keep
working.

The mode binning delegates to the canonical core
:func:`oneuniverse.simulation.validation.binned_mode_powers` (S9); this module
keeps only the twin k-convention (fundamental-mode-spaced edges ``kf/2 + i·kf``)
and the NaN-for-empty-bins presentation.

- r(k)        : cross-correlation coefficient (phases).
- transfer    : T(k) = ⟨rec·truth⟩ / ⟨truth²⟩ (amplitude recovery).
- power_ratio : P_a(k) / P_b(k).
- k_half      : smallest k where r(k) < 0.5 (the reconstruction scale; nan if
  recovered to the Nyquist frequency).
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class RecoveryMetrics:
    k: np.ndarray
    r: np.ndarray
    transfer: np.ndarray
    power_ratio: np.ndarray
    k_half: float


def _transfer(rec, truth, box_size):
    # Delegates to the canonical mode-binning core; same kf edges,
    # T(k) = S_ab/S_bb identical to the pre-consolidation sums.
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
