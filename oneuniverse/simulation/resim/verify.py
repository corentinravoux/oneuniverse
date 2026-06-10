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
    """Binned cross-correlation r(k) of two equal-shape real fields.

    S9: delegates to the canonical mode-binning core
    (:func:`oneuniverse.simulation.validation.binned_mode_powers`); keeps the
    gate convention (linear edges over the populated |k| range; empty or
    zero-power bins skipped). Numerics identical to the previous inline code.
    """
    from oneuniverse.simulation.validation import binned_mode_powers
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    kx = np.fft.fftfreq(n, d=box_size / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2).ravel()
    edges = np.linspace(kmag[kmag > 0].min(), kmag.max(), n_bins + 1)
    centres, S_aa, S_bb, S_ab, n_modes = binned_mode_powers(
        a, b, box=box_size, edges=edges)
    keep = (n_modes > 0) & (S_aa > 0) & (S_bb > 0)
    return centres[keep], S_ab[keep] / np.sqrt(S_aa[keep] * S_bb[keep])


def gate1_consistency(mini_ic: np.ndarray, parent_sub: np.ndarray, *,
                      box_size: float) -> Dict:
    """Pre-run check that the mini IC matches the parent on the sub-volume."""
    k, r = _cross_r(mini_ic, parent_sub, box_size)
    cell_corr = float(np.corrcoef(mini_ic.ravel(), parent_sub.ravel())[0, 1])
    low = r[k < np.median(k)]
    passed = bool(cell_corr > 0.95 and np.nanmedian(low) > 0.9)
    return {"k": k, "r": r, "cell_corr": cell_corr,
            "r_lowk": float(np.nanmedian(low)), "passed": passed}


def gate2_dynamical(inner: np.ndarray, reference: np.ndarray, *,
                    box_size: float) -> Dict:
    """Post-run (sufficient) check: the resimulated inner region matches the
    full-box reference on the shared volume after evolution. The empirical
    feasibility verdict — its large-scale ``r_lowk`` is the headline number.
    """
    k, r = _cross_r(inner, reference, box_size)
    cell_corr = float(np.corrcoef(inner.ravel(), reference.ravel())[0, 1])
    low = r[k < np.median(k)]
    r_lowk = float(np.nanmedian(low))
    return {"k": k, "r": r, "cell_corr": cell_corr, "r_lowk": r_lowk,
            "passed": bool(r_lowk > 0.8)}
