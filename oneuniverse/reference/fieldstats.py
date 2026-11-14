"""The single P(k) / r(k) estimator of ``oneuniverse.reference``.

Conventions
-----------
* Fields are real cubic ``(n, n, n)`` grids of side ``box`` (Mpc/h); ``k`` in
  h/Mpc, ``P`` in (Mpc/h)³. Full complex FFT, ``P(k) = |δ_k|²·V/N²``
  (``δ_k = fftn(δ)``, ``N = n³``, ``V = box³``), the normalisation of
  :mod:`oneuniverse.reference.linear`. Cross power ``P_ab = Re(a_k b_k*)·V/N²``.
* Binning (:func:`bin_edges_kf`, the one binning): edges ``k_f·(j + 1/2)``,
  ``j = 0, 1, …``, ``k_f = 2π/box``, the last edge being the largest such
  value ``≤ k_N = π·n/box``. Bins are therefore centred on integer multiples
  of ``k_f``; the k = 0 mode and modes above the last edge (the Nyquist shell
  and grid corners) are excluded. Bins containing no mode are dropped from
  every output, so ``k`` is strictly increasing with ``k[0] > 0``.
* Per bin, ``k`` is the mean ``|k|`` of its modes and ``P`` the mean of the
  per-mode powers. Ratios (``r``, ``T``, power ratio) are ratios of bin
  means; a bin with a zero denominator yields NaN.
* ``cross_r = P_ab / sqrt(P_aa·P_bb)``; ``transfer T = P_ab(rec, truth) /
  P_truth`` (amplitude of ``rec`` projected on ``truth``, as in the old
  ``twin.metrics``); ``power_ratio = P_rec / P_truth``.
* :func:`k_half`: first ``k`` where ``r`` drops strictly below ``level``,
  linearly interpolated between the two bracketing bins; ``k[0]`` if ``r``
  starts below ``level``; ``k[-1]`` if it never drops below. NaN entries are
  ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.linear import k_grid

__all__ = [
    "RecoveryMetrics",
    "bin_edges_kf",
    "cell_corr",
    "cross_power",
    "cross_r",
    "k_half",
    "power_spectrum",
    "recover_metrics",
    "transfer",
]


def bin_edges_kf(n_grid: int, box: float) -> np.ndarray:
    """Fundamental-spaced edges ``k_f·(j + 1/2)`` up to the Nyquist ``π·n/box``."""
    n = int(n_grid)
    if n != n_grid or n < 3:
        raise ValidationError(f"n_grid must be an integer >= 3, got {n_grid}")
    box = float(box)
    if not np.isfinite(box) or box <= 0:
        raise ValidationError(f"box must be a positive finite length, got {box}")
    kf = 2.0 * np.pi / box
    return kf * (np.arange((n + 1) // 2) + 0.5)


@lru_cache(maxsize=16)
def _binning(n: int, box: float):
    edges = bin_edges_kf(n, box)
    _, _, _, kmag = k_grid(n, box)
    idx = np.digitize(kmag.ravel(), edges) - 1
    nb = len(edges) - 1
    valid = (idx >= 0) & (idx < nb)
    idx = idx[valid]
    counts = np.bincount(idx, minlength=nb)
    ksum = np.bincount(idx, weights=kmag.ravel()[valid], minlength=nb)
    keep = counts > 0
    for arr in (valid, idx, counts):
        arr.flags.writeable = False
    k = ksum[keep] / counts[keep]
    k.flags.writeable = False
    return valid, idx, counts, keep, k


def _field(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3 or not (arr.shape[0] == arr.shape[1] == arr.shape[2]):
        raise ValidationError(f"{name} must be a cubic (n, n, n) grid, got shape {arr.shape}")
    return arr


def _bin(per_mode: np.ndarray, n: int, box: float) -> tuple[np.ndarray, np.ndarray]:
    valid, idx, counts, keep, k = _binning(n, box)
    s = np.bincount(idx, weights=per_mode.ravel()[valid], minlength=len(counts))
    return k.copy(), s[keep] / counts[keep]


def _spectra(a, b, box: float):
    """``(k, P_aa, P_bb, P_ab)`` binned on :func:`bin_edges_kf`."""
    a, b = _field(a, "a"), _field(b, "b")
    if a.shape != b.shape:
        raise ValidationError(f"fields differ in shape: {a.shape} vs {b.shape}")
    n = a.shape[0]
    norm = float(box) ** 3 / float(n) ** 6
    ak, bk = np.fft.fftn(a), np.fft.fftn(b)
    k, paa = _bin((ak.real ** 2 + ak.imag ** 2) * norm, n, box)
    _, pbb = _bin((bk.real ** 2 + bk.imag ** 2) * norm, n, box)
    _, pab = _bin((ak * bk.conj()).real * norm, n, box)
    return k, paa, pbb, pab


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    good = den != 0
    out[good] = num[good] / den[good]
    return out


def power_spectrum(delta, *, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Binned auto power ``(k, P)`` of a real cubic field."""
    d = _field(delta, "delta")
    n = d.shape[0]
    dk = np.fft.fftn(d)
    return _bin((dk.real ** 2 + dk.imag ** 2) * (float(box) ** 3 / float(n) ** 6), n, box)


def cross_power(a, b, *, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Binned cross power ``(k, P_ab)``, ``P_ab = Re(a_k b_k*)·V/N²``."""
    k, _, _, pab = _spectra(a, b, box)
    return k, pab


def cross_r(a, b, *, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation coefficient ``r(k) = P_ab / sqrt(P_aa·P_bb)``."""
    k, paa, pbb, pab = _spectra(a, b, box)
    return k, _ratio(pab, np.sqrt(paa * pbb))


def transfer(rec, truth, *, box: float) -> tuple[np.ndarray, np.ndarray]:
    """Transfer ``T(k) = P_cross(rec, truth) / P_truth``."""
    k, _, ptt, pab = _spectra(rec, truth, box)
    return k, _ratio(pab, ptt)


def k_half(k, r, level: float = 0.5) -> float:
    """First ``k`` where ``r`` drops strictly below ``level`` (linear interpolation)."""
    k = np.asarray(k, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if k.shape != r.shape or k.ndim != 1:
        raise ValidationError(f"k and r must be 1-D of equal length, got {k.shape}, {r.shape}")
    good = np.isfinite(k) & np.isfinite(r)
    k, r = k[good], r[good]
    if k.size == 0:
        raise ValidationError("k_half needs at least one finite (k, r) pair")
    below = np.flatnonzero(r < level)
    if below.size == 0:
        return float(k[-1])
    i = int(below[0])
    if i == 0:
        return float(k[0])
    k0, k1, r0, r1 = k[i - 1], k[i], r[i - 1], r[i]
    return float(k0 + (level - r0) * (k1 - k0) / (r1 - r0))


def cell_corr(a, b) -> float:
    """Pearson correlation coefficient of the flattened cells of ``a`` and ``b``."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValidationError(f"fields differ in size: {a.size} vs {b.size}")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class RecoveryMetrics:
    """Binned recovery metrics of a reconstructed field against the truth."""

    k: np.ndarray
    r: np.ndarray
    transfer: np.ndarray
    power_ratio: np.ndarray
    k_half: float


def recover_metrics(rec, truth, *, box: float) -> RecoveryMetrics:
    """``r(k)``, ``T(k)``, ``P_rec/P_truth`` and ``k_half(r, 0.5)`` of ``rec`` vs ``truth``."""
    k, prr, ptt, pab = _spectra(rec, truth, box)
    r = _ratio(pab, np.sqrt(prr * ptt))
    return RecoveryMetrics(k=k, r=r, transfer=_ratio(pab, ptt),
                           power_ratio=_ratio(prr, ptt), k_half=k_half(k, r))
