"""Field-validation estimators — the proper way to compare two density fields.

Given a candidate field ``a`` (e.g. a resimulation) and a reference ``b``
(e.g. the full simulation), on the same grid, the standard cosmological
field-level diagnostics are:

- **cross-correlation** r(k) = P_ab / √(P_aa P_bb) — *phase* agreement (right
  structure in the right place), amplitude-independent. r=1 is perfect.
- **transfer / propagator** T(k) = P_ab / P_bb — *amplitude* recovery
  (δ_a ≈ T(k)·δ_b + noise). T=1 is perfect.
- **power ratio** P_aa / P_bb — total power match.
- **stochasticity** S(k) = 1 − r²(k) — the fraction of the candidate's
  variance that is **not** predictable from the reference (the irreducible
  "noise" of the comparison). S=0 is perfect.
- **k_half** — the scale where r(k) drops to 0.5 (the agreement scale).
- the **1-point PDF** of the two fields — non-Gaussian / amplitude check
  beyond two-point statistics.

All pure numpy. Decomposition: P_aa = T²·P_bb + P_noise, with
T = r·√(P_aa/P_bb) and P_noise/P_aa = 1 − r².
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FieldValidation:
    k: np.ndarray
    r: np.ndarray              # cross-correlation r(k)
    transfer: np.ndarray       # T(k) = P_ab / P_bb
    power_ratio: np.ndarray    # P_aa / P_bb
    stochasticity: np.ndarray  # 1 - r^2(k)
    k_half: float              # k where r(k) = 0.5
    pdf_edges: np.ndarray
    pdf_a: np.ndarray
    pdf_b: np.ndarray
    var_a: float
    var_b: float


def binned_mode_powers(a: np.ndarray, b: np.ndarray, *, box: float,
                       edges: np.ndarray):
    """**The canonical mode-binning core (S9).** Per-|k|-bin *summed* mode
    powers of two real fields on the same grid:

        (centres, S_aa, S_bb, S_ab, n_modes)

    Full-length arrays aligned with ``centres = 0.5*(edges[:-1]+edges[1:])``;
    empty bins carry zero sums and ``n_modes == 0`` (callers decide NaN vs
    skip). Every field-validation estimator in the package — `validate_field`
    here, `twin.verify.cross_correlation/power_ratio`,
    `twin.validation.recover_metrics`, `resim.verify` gates — delegates to
    this single binning, so one k-convention bug cannot fork four ways.
    All the usual ratios (r, T, P_a/P_b) are sums-ratios, identical to
    means-ratios since the mode counts cancel.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = a.shape[0]
    ak = np.fft.rfftn(a); bk = np.fft.rfftn(b)
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2).ravel()
    paa = (np.abs(ak) ** 2).ravel()
    pbb = (np.abs(bk) ** 2).ravel()
    pab = np.real(ak * np.conj(bk)).ravel()
    edges = np.asarray(edges, float)
    idx = np.digitize(kmag, edges)
    nb = len(edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    S_aa = np.zeros(nb); S_bb = np.zeros(nb); S_ab = np.zeros(nb)
    n_modes = np.zeros(nb, dtype=int)
    for i in range(1, len(edges)):
        m = idx == i
        c = int(m.sum())
        if c:
            n_modes[i - 1] = c
            S_aa[i - 1] = paa[m].sum()
            S_bb[i - 1] = pbb[m].sum()
            S_ab[i - 1] = pab[m].sum()
    return centres, S_aa, S_bb, S_ab, n_modes


def _binned_powers(a: np.ndarray, b: np.ndarray, box: float, n_bins: int):
    """Return (k, P_aa, P_bb, P_ab) binned in |k| (means; empty bins skipped)."""
    n = a.shape[0]
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2).ravel()
    pos = kmag > 0
    edges = np.linspace(kmag[pos].min(), kmag.max(), n_bins + 1)
    centres, S_aa, S_bb, S_ab, n_modes = binned_mode_powers(
        a, b, box=box, edges=edges)
    keep = n_modes > 0
    nm = n_modes[keep]
    return (centres[keep], S_aa[keep] / nm, S_bb[keep] / nm, S_ab[keep] / nm)


def _k_half(k: np.ndarray, r: np.ndarray) -> float:
    below = np.where(r < 0.5)[0]
    return float(k[below[0]]) if len(below) else float("nan")


def validate_field(field: np.ndarray, reference: np.ndarray, *, box: float,
                   n_bins: int = 20, pdf_range: Tuple[float, float] = (-3.0, 6.0),
                   pdf_bins: int = 50) -> FieldValidation:
    """Full field-validation diagnostics of ``field`` against ``reference``."""
    a = np.asarray(field, float); b = np.asarray(reference, float)
    k, Paa, Pbb, Pab = _binned_powers(a, b, box, n_bins)
    r = Pab / np.sqrt(Paa * Pbb)
    transfer = Pab / Pbb
    power_ratio = Paa / Pbb
    stoch = np.clip(1.0 - r ** 2, 0.0, None)
    edges = np.linspace(*pdf_range, pdf_bins + 1)
    pdf_a, _ = np.histogram(a.ravel(), bins=edges, density=True)
    pdf_b, _ = np.histogram(b.ravel(), bins=edges, density=True)
    return FieldValidation(
        k=k, r=r, transfer=transfer, power_ratio=power_ratio,
        stochasticity=stoch, k_half=_k_half(k, r), pdf_edges=edges,
        pdf_a=pdf_a, pdf_b=pdf_b, var_a=float(a.var()), var_b=float(b.var()),
    )
