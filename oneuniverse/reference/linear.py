"""Linear theory for the reference engine: P(k), growth, Gaussian and Zel'dovich fields.

Conventions (every later ``reference`` module relies on them)
--------------------------------------------------------------
Units
    Wavenumbers ``k`` in h/Mpc, power ``P`` in (Mpc/h)³, positions and box
    sides in Mpc/h, velocities in km/s (peculiar). ``H(a) = 100·E(a)·h``
    km/s/Mpc; in Mpc/h units the factor ``h`` cancels, so a displacement ψ in
    Mpc/h moves at ``a·H(a)·f·ψ = 100·a·E(a)·f·ψ`` km/s.
Cosmology
    Flat ΛCDM only: ``E(a)² = Ωm a⁻³ + (1 − Ωm)``; radiation is neglected.
    Any ``CosmologySpec`` with ``w0 != -1`` or ``wa != 0`` raises
    :class:`~oneuniverse._core.errors.ValidationError`.
Growth
    ``D(a) ∝ (5Ωm/2)·E(a)·I(a)``, ``I(a) = ∫₀ᵃ da'/(a'E(a'))³`` (the exact
    growing mode for ΛCDM), normalised so ``D(z=0) = 1`` exactly. The growth
    rate is ``f = dlnD/dlna = dlnE/dlna + 1/(a² E³ I(a))`` with
    ``dlnE/dlna = −(3/2) Ωm a⁻³ / E²``: the same integral as ``D``, not the
    γ = 0.55 approximation.
Power spectrum
    Eisenstein & Hu (1998) *no-wiggle* transfer function (eqs. 26–31, no BAO
    wiggles), primordial ``k^n_s``, normalised to ``cosmo.sigma8`` with a
    real-space top-hat of radius 8 Mpc/h. The amplitude is cached per
    ``CosmologySpec`` (a frozen, hashable dataclass). CMB temperature 2.7255 K.
Grid
    ``n_grid`` cells per side; cell ``i`` covers ``[i·cell, (i+1)·cell)``,
    ``cell = box/n_grid``; field values and Lagrangian positions ``q`` sit at
    cell centres ``(i + 0.5)·cell``. Arrays are indexed ``[ix, iy, iz]``;
    flattened particle arrays follow C order of that grid.
Fourier
    Full complex FFT (``np.fft.fftn``, not rfft). :func:`k_grid` returns
    ``2π·np.fft.fftfreq(n, d=cell)`` per axis. Power normalisation:
    ``P(k) = |δ_k|²·V/N²`` with ``δ_k = fftn(δ)``, ``N = n³``, ``V = box³``.
    :func:`gaussian_field` draws real-space white noise ``w`` and sets
    ``δ_k = fftn(w)·sqrt(P(|k|)·N/V)`` with the k = 0 mode zeroed; the
    multiplier depends on ``|k|`` only, so Hermitian symmetry (Nyquist planes
    included) is preserved and ``ifftn`` is real. Hence
    ``var(δ) ≈ Σ_k P(k)/V`` over the grid modes.
Zel'dovich
    ``δ = −∇·ψ`` → ``ψ_k = i k δ_k / k²``. The derivative along an axis is
    zero on that axis' Nyquist plane (its ``i k`` is not Hermitian there), so
    ψ is exactly real. ``pos = q + D(z)·ψ`` wrapped into ``[0, box)``;
    ``vel = 100·a·E(a)·f(z)·D(z)·ψ`` km/s, with ψ computed from the z = 0
    linear field. This fixes the old velocity, which used ``100·h·f/(1+z)``
    (off by ``E(a)·h``).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyarrow as pa
from scipy import integrate

from oneuniverse._core.errors import ValidationError
from oneuniverse.simulation.specs import CosmologySpec

__all__ = [
    "eh98_transfer",
    "find_peaks",
    "gaussian_field",
    "growth_factor",
    "growth_rate",
    "k_grid",
    "linear_power",
    "zeldovich",
]

T_CMB = 2.7255  # K
_SIGMA8_R = 8.0  # Mpc/h


def _require_lcdm(cosmo: CosmologySpec) -> None:
    if cosmo.w0 != -1.0 or cosmo.wa != 0.0:
        raise ValidationError(
            "reference.linear supports flat ΛCDM only (w0 = -1, wa = 0); "
            f"got w0={cosmo.w0}, wa={cosmo.wa}")


def _check_grid(n_grid: int, box: float) -> tuple[int, float]:
    n = int(n_grid)
    if n != n_grid or n < 2:
        raise ValidationError(f"n_grid must be an integer >= 2, got {n_grid}")
    box = float(box)
    if not np.isfinite(box) or box <= 0:
        raise ValidationError(f"box must be a positive finite length, got {box}")
    return n, box


def _cubic(field, name: str = "delta") -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3 or not (arr.shape[0] == arr.shape[1] == arr.shape[2]):
        raise ValidationError(f"{name} must be a cubic (n, n, n) grid, got shape {arr.shape}")
    return arr


# --------------------------------------------------------------------------
# Power spectrum
# --------------------------------------------------------------------------

def eh98_transfer(k, cosmo: CosmologySpec) -> np.ndarray:
    """Eisenstein & Hu (1998) no-wiggle transfer function ``T(k)``; ``k`` in h/Mpc."""
    k = np.asarray(k, dtype=np.float64)
    h, om, ob = cosmo.h, cosmo.omega_m, cosmo.omega_b
    omh2, obh2 = om * h * h, ob * h * h
    fb = ob / om
    theta = T_CMB / 2.7
    # sound horizon (EH98 eq. 26), Mpc -> Mpc/h
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75) * h
    alpha = 1.0 - 0.328 * np.log(431.0 * omh2) * fb + 0.38 * np.log(22.3 * omh2) * fb ** 2
    # k·s is dimensionless in either unit system
    gamma_eff = om * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def _shape(k, cosmo: CosmologySpec) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return k ** cosmo.n_s * eh98_transfer(k, cosmo) ** 2


def _top_hat(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = x > 1e-6
    xn = x[nz]
    out[nz] = 3.0 * (np.sin(xn) - xn * np.cos(xn)) / xn ** 3
    return out


@lru_cache(maxsize=64)
def _amplitude(cosmo: CosmologySpec) -> float:
    """``A`` such that ``A·k^n_s·T²`` has σ(8 Mpc/h) = sigma8 (integrated in ln k)."""
    lnk = np.linspace(np.log(1e-6), np.log(1e3), 40001)
    k = np.exp(lnk)
    integrand = k ** 3 * _shape(k, cosmo) * _top_hat(k * _SIGMA8_R) ** 2 / (2.0 * np.pi ** 2)
    sigma2 = integrate.simpson(integrand, x=lnk)
    return float(cosmo.sigma8 ** 2 / sigma2)


def linear_power(k, cosmo: CosmologySpec, z: float = 0.0) -> np.ndarray:
    """σ8-normalised linear matter ``P(k, z) = A·k^n_s·T(k)²·D(z)²`` in (Mpc/h)³."""
    _require_lcdm(cosmo)
    d = growth_factor(z, cosmo)
    return _amplitude(cosmo) * _shape(k, cosmo) * d * d


# --------------------------------------------------------------------------
# Growth (flat ΛCDM, exact integral)
# --------------------------------------------------------------------------

def _e(a: float, om: float) -> float:
    return float(np.sqrt(om * a ** -3 + (1.0 - om)))


@lru_cache(maxsize=4096)
def _growth_integral(a: float, om: float) -> float:
    """``I(a) = ∫₀ᵃ da'/(a'E)³ = ∫₀ᵃ a'^{3/2} / (Ωm + ΩΛ a'³)^{3/2} da'``."""
    ol = 1.0 - om
    val, _ = integrate.quad(lambda x: x ** 1.5 / (om + ol * x ** 3) ** 1.5, 0.0, a,
                            epsabs=0.0, epsrel=1e-12, limit=200)
    return float(val)


def _growth_unnorm(a: float, om: float) -> float:
    return 2.5 * om * _e(a, om) * _growth_integral(a, om)


def _scale_factor(z) -> float:
    z = float(z)
    if not np.isfinite(z) or z <= -1.0:
        raise ValidationError(f"redshift must be finite and > -1, got {z}")
    return 1.0 / (1.0 + z)


def growth_factor(z: float, cosmo: CosmologySpec) -> float:
    """Linear growth ``D(z)/D(0)`` for flat ΛCDM; ``growth_factor(0.0) == 1.0`` exactly."""
    _require_lcdm(cosmo)
    a = _scale_factor(z)
    return _growth_unnorm(a, cosmo.omega_m) / _growth_unnorm(1.0, cosmo.omega_m)


def growth_rate(z: float, cosmo: CosmologySpec) -> float:
    """``f = dlnD/dlna = −(3/2)Ωm a⁻³/E² + 1/(a² E³ I(a))`` from the growth integral."""
    _require_lcdm(cosmo)
    a = _scale_factor(z)
    om = cosmo.omega_m
    e = _e(a, om)
    return float(-1.5 * om * a ** -3 / e ** 2 + 1.0 / (a * a * e ** 3 * _growth_integral(a, om)))


# --------------------------------------------------------------------------
# Grids and fields
# --------------------------------------------------------------------------

def k_grid(n_grid: int, box: float):
    """Full-FFT wavenumbers (h/Mpc) of an ``n_grid³`` mesh of side ``box``.

    Returns ``(kx, ky, kz, kmag)``: ``kx``, ``ky``, ``kz`` are
    ``2π·fftfreq(n, d=box/n)`` shaped ``(n,1,1)``, ``(1,n,1)``, ``(1,1,n)`` so
    they broadcast against ``(n, n, n)`` arrays; ``kmag`` is the full
    ``(n, n, n)`` modulus.
    """
    n, box = _check_grid(n_grid, box)
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kx, ky, kz = k1[:, None, None], k1[None, :, None], k1[None, None, :]
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    return kx, ky, kz, kmag


def gaussian_field(cosmo: CosmologySpec, *, box: float, n_grid: int, z: float = 0.0,
                   seed: int) -> np.ndarray:
    """Gaussian linear density contrast on an ``n_grid³`` mesh (float64, mean exactly 0).

    White noise in real space (``np.random.default_rng(seed)``), coloured by
    ``sqrt(P(|k|, z)·N/V)`` in Fourier space with the k = 0 mode zeroed, so
    the estimator ``|δ_k|²·V/N²`` averages ``P`` and ``var(δ) ≈ Σ_k P/V``.
    """
    n, box = _check_grid(n_grid, box)
    _, _, _, kmag = k_grid(n, box)
    pk = np.zeros_like(kmag)
    nz = kmag > 0.0
    pk[nz] = linear_power(kmag[nz], cosmo, z=z)
    white = np.random.default_rng(seed).standard_normal((n, n, n))
    delta_k = np.fft.fftn(white) * np.sqrt(pk * (n ** 3) / box ** 3)
    delta = np.fft.ifftn(delta_k).real
    delta -= delta.mean()
    return np.ascontiguousarray(delta, dtype=np.float64)


def _displacement(delta: np.ndarray, box: float) -> np.ndarray:
    """ψ with ``−∇·ψ = δ`` (Nyquist-plane derivatives zeroed), shape ``(n³, 3)``."""
    n = delta.shape[0]
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    _, _, _, kmag = k_grid(n, box)
    k2 = kmag ** 2
    k2[0, 0, 0] = 1.0
    delta_k = np.fft.fftn(delta)
    delta_k[0, 0, 0] = 0.0
    phi_k = delta_k / k2
    kd = k1.copy()
    if n % 2 == 0:
        kd[n // 2] = 0.0
    shapes = ((n, 1, 1), (1, n, 1), (1, 1, n))
    psi = np.empty((n ** 3, 3), dtype=np.float64)
    for i, shp in enumerate(shapes):
        psi[:, i] = np.fft.ifftn(1j * kd.reshape(shp) * phi_k).real.ravel()
    return psi


def zeldovich(delta_z0, cosmo: CosmologySpec, *, box: float, z: float):
    """Zel'dovich particles from the z = 0 linear field ``delta_z0``.

    Returns ``(pos, vel)``, each ``(n³, 3)`` float64 in C order of the
    Lagrangian grid: ``pos = q + D(z)·ψ`` (Mpc/h, wrapped into ``[0, box)``)
    and ``vel = 100·a·E(a)·f(z)·D(z)·ψ`` (km/s, peculiar).
    """
    delta = _cubic(delta_z0, "delta_z0")
    n, box = _check_grid(delta.shape[0], box)
    _require_lcdm(cosmo)
    a = _scale_factor(z)
    d, f = growth_factor(z, cosmo), growth_rate(z, cosmo)
    e = _e(a, cosmo.omega_m)
    psi = _displacement(delta, box)
    g = (np.arange(n) + 0.5) * (box / n)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    pos = np.mod(q + d * psi, box)
    pos[pos >= box] -= box
    vel = (100.0 * a * e * f * d) * psi
    return pos, vel


# --------------------------------------------------------------------------
# Peaks
# --------------------------------------------------------------------------

_PEAK_SCHEMA = pa.schema([(c, pa.float64()) for c in ("x", "y", "z", "height", "mass_proxy")])


def _rolled(d: np.ndarray):
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                yield (sx, sy, sz), np.roll(d, (sx, sy, sz), axis=(0, 1, 2))


def find_peaks(delta, *, box: float, threshold: float = 2.0) -> pa.Table:
    """Toy halo catalogue: periodic local maxima of ``delta`` above ``threshold·σ_δ``.

    A peak is a cell strictly greater than all 26 periodic neighbours with
    ``δ/σ_δ > threshold`` (``σ_δ`` = standard deviation of ``delta``).
    Columns (float64): ``x, y, z`` cell-centre coordinates in Mpc/h;
    ``height = δ/σ_δ``; ``mass_proxy`` = sum of ``max(δ, 0)`` over the 3³
    periodic neighbourhood (peak included) × cell volume, in (Mpc/h)³ — a
    toy proxy, not a halo mass.
    """
    d = _cubic(delta)
    n, box = _check_grid(d.shape[0], box)
    sigma = float(d.std())
    if sigma == 0.0:
        return _PEAK_SCHEMA.empty_table()
    is_peak = np.ones(d.shape, dtype=bool)
    pos_sum = np.zeros_like(d)
    dpos = np.maximum(d, 0.0)
    for shift, rolled in _rolled(d):
        if shift != (0, 0, 0):
            is_peak &= d > rolled
    for _, rolled in _rolled(dpos):
        pos_sum += rolled
    sel = is_peak & (d / sigma > threshold)
    idx = np.argwhere(sel)
    cell = box / n
    centres = (idx + 0.5) * cell
    return pa.table({
        "x": centres[:, 0], "y": centres[:, 1], "z": centres[:, 2],
        "height": d[sel] / sigma,
        "mass_proxy": pos_sum[sel] * cell ** 3,
    }, schema=_PEAK_SCHEMA)
