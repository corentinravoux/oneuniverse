"""Fast particle-mesh (PM) N-body for the reference engine.

Reuses every convention of :mod:`oneuniverse.reference.linear` (grid, full
complex FFT, Nyquist planes, units, flat ΛCDM ``E(a)``, growth, Zel'dovich).

Conventions
-----------
Mesh and CIC
    ``n_grid`` cells per side, cell ``i`` centred at ``(i + 0.5)·cell``.
    :func:`deposit_cic` returns particle *counts* per cell (unit mass per
    particle): a particle at a cell centre lands entirely in that cell, one on
    a cell face splits 1/2–1/2. Positions may be anywhere; they are wrapped
    into ``[0, box)`` (periodic) before depositing. Weights are accumulated
    with ``np.bincount`` on flattened cell indices (eight corner passes, no
    per-particle Python loop). :func:`interpolate_cic` uses the same kernel and
    is its exact adjoint: ``Σ_p w_p·interp(g)(x_p) = Σ_cells g·deposit(w)``.
Poisson
    :func:`pm_force` solves ``∇²φ = δ`` (dimensionless source, ``φ_k =
    −δ_k/k²``, k = 0 mode zeroed) and returns the three grids of
    **∇φ** (``(∇φ)_k = i k φ_k``), the derivative along an axis being zero on
    that axis' Nyquist plane for even ``n`` (as in ``linear``). This sign is
    fixed by the single-mode test: ``δ = sin(kx)`` gives ``∂ₓφ =
    −cos(kx)/k``. The *attractive* acceleration is ``−∇φ``; the cosmological
    prefactor is applied by :func:`run_pm`, which kicks with
    ``dp/da = −(3/2)·Ωm·∇φ/(a²·E(a))``.
    ``deconvolve=True`` divides ``δ_k`` by one CIC window
    ``Π_i sinc²(k_i·cell/2)`` (off by default: the reference runs are
    validated on large scales, where the window is ≈ 1).
Isolated Poisson
    :func:`pm_force_isolated` is the Hockney–Eastwood open-boundary solve:
    ``δ`` is embedded in a zero-padded ``(2n)³`` grid and cyclically convolved
    with the free-space Green's-function gradient ``∇G(r) = r/(4π r³)``
    (``G = −1/(4πr)`` solves ``∇²G = δ_D``), sampled at the minimum-image cell
    offsets of the padded grid with ``∇G(0) = 0`` (no self-force); the
    result is cropped back to ``n³``. Offsets up to ``±(n − 1)`` cells are
    represented without aliasing, so periodic images are absent. The gradient
    kernel is sampled directly (rather than differentiating a sampled ``G``
    spectrally), which keeps the cropped region free of wrap-around
    artefacts. Same sign/units as :func:`pm_force` (returns ∇φ).
Time variable and momentum
    Units ``H0 = 1`` and lengths in Mpc/h. Time variable ``a``; canonical
    momentum ``p = a²·dx/dt = a³·E(a)·dx/da`` (``x`` comoving). Equations:
    ``dx/da = p/(a³E)``, ``dp/da = −(3/2)Ωm·∇φ_δ/(a²E)``. Peculiar velocity
    ``v = a·dx/dt = p/a`` in units of ``H0·(Mpc/h) = 100 km/s``, so the
    public ``vel`` arrays (input and output, km/s, the units of
    :func:`~oneuniverse.reference.linear.zeldovich`) are ``vel = 100·p/a``.
    The Zel'dovich growing mode ``x = q + D·ψ`` has ``p = a²·E·f·D·ψ``, i.e.
    ``vel = 100·a·E·f·D·ψ`` exactly as ``linear.zeldovich``.
Stepping
    Kick–drift–kick leapfrog with ``n_steps`` steps **uniform in a** (the old
    ``simulation.pm.run`` choice, kept). Per step ``[a0, a1]``, ``ah =
    (a0 + a1)/2``: kick by ``K(a0, ah)``, drift by ``Dr(a0, a1)``, recompute
    the force, kick by ``K(ah, a1)``, with ``K = ∫ da/(a²E)`` and
    ``Dr = ∫ da/(a³E)`` computed by ``scipy.integrate.quad`` (the old code used
    a 64-point trapezoid). ``E(a)`` is ``linear``'s flat-ΛCDM expansion rate.
    The force uses ``δ = ρ/ρ̄ − 1`` of the CIC-deposited particles on the
    ``n_grid³`` mesh.
Snapshots
    For each requested ``z`` in ``[z_end, z_start]`` the run reports δ at
    exactly ``a_s = 1/(1+z)``: inside the step containing ``a_s``, after the
    first half kick, positions are drifted from ``a0`` to ``a_s`` with the
    half-step momentum (a side copy; the main integration is untouched) and
    deposited. ``z_start`` gives the initial δ, ``z_end`` the final one. Keys
    of ``out["snapshots"]`` are the floats as given.
Force modes
    ``force="periodic"`` uses :func:`pm_force`; ``"isolated"`` uses
    :func:`pm_force_isolated` on the same mesh (deposit and position wrapping
    stay periodic: the caller keeps particles inside the box, as the buffered
    resimulation does).
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy import integrate

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.linear import (
    _check_grid,
    _cubic,
    _e,
    _require_lcdm,
    _scale_factor,
    gaussian_field,
    k_grid,
    zeldovich,
)
from oneuniverse.simulation.specs import CosmologySpec

__all__ = [
    "deposit_cic",
    "interpolate_cic",
    "pm_force",
    "pm_force_isolated",
    "run_pm",
    "zeldovich_pm_ic",
    "zeldovich_pm_ic_from_field",
]

_FORCES = ("periodic", "isolated")


# --------------------------------------------------------------------------
# CIC
# --------------------------------------------------------------------------

def _positions(pos, name: str = "pos") -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValidationError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must be finite")
    return arr


def _cic_corners(pos: np.ndarray, n: int, box: float):
    """Yield ``(flat_index, weight)`` for the 8 CIC corners of every particle."""
    g = np.mod(pos, box) * (n / box) - 0.5        # cell-centre coordinates
    i0 = np.floor(g)
    d = g - i0
    i0 = i0.astype(np.int64) % n
    i1 = (i0 + 1) % n
    idx = (i0, i1)
    w = (1.0 - d, d)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                flat = (idx[a][:, 0] * n + idx[b][:, 1]) * n + idx[c][:, 2]
                yield flat, w[a][:, 0] * w[b][:, 1] * w[c][:, 2]


def deposit_cic(pos, n_grid: int, box: float, weights=None) -> np.ndarray:
    """CIC counts of particles on an ``n_grid³`` periodic mesh (float64).

    ``weights`` (optional, shape ``(N,)``) replaces the unit particle mass.
    """
    n, box = _check_grid(n_grid, box)
    p = _positions(pos)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(p),):
            raise ValidationError(f"weights must have shape ({len(p)},), got {weights.shape}")
    out = np.zeros(n ** 3, dtype=np.float64)
    for flat, w in _cic_corners(p, n, box):
        out += np.bincount(flat, weights=w if weights is None else w * weights,
                           minlength=n ** 3)
    return out.reshape(n, n, n)


def interpolate_cic(grid, pos, box: float) -> np.ndarray:
    """Sample a periodic mesh at particle positions with the CIC kernel (adjoint of deposit)."""
    g = _cubic(grid, "grid")
    n, box = _check_grid(g.shape[0], box)
    p = _positions(pos)
    flat_grid = g.ravel()
    out = np.zeros(len(p), dtype=np.float64)
    for flat, w in _cic_corners(p, n, box):
        out += flat_grid[flat] * w
    return out


# --------------------------------------------------------------------------
# Poisson
# --------------------------------------------------------------------------

def _cic_window(n: int, box: float) -> np.ndarray:
    cell = box / n
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=cell)
    w1 = np.sinc(k1 * cell / (2.0 * np.pi)) ** 2      # np.sinc(x) = sin(πx)/(πx)
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def pm_force(delta, box: float, *, deconvolve: bool = False):
    """Periodic FFT Poisson: grids ``(∂ₓφ, ∂ᵧφ, ∂_zφ)`` of ``∇²φ = δ``.

    Returns ∇φ (the attraction is ``−∇φ``); see the module docstring.
    """
    d = _cubic(delta)
    n, box = _check_grid(d.shape[0], box)
    kx, ky, kz, kmag = k_grid(n, box)
    k2 = kmag ** 2
    k2[0, 0, 0] = 1.0
    delta_k = np.fft.fftn(d)
    if deconvolve:
        delta_k /= _cic_window(n, box)
    phi_k = -delta_k / k2
    phi_k[0, 0, 0] = 0.0
    out = []
    for kc in (kx, ky, kz):
        kd = kc.copy()
        if n % 2 == 0:
            kd[_nyq_index(kd.shape, n)] = 0.0
        out.append(np.fft.ifftn(1j * kd * phi_k).real)
    return tuple(out)


def _nyq_index(shape: tuple[int, ...], n: int) -> tuple[int, ...]:
    """Index of the Nyquist entry of a broadcast 1-D wavenumber array."""
    return tuple(n // 2 if s == n else 0 for s in shape)


@lru_cache(maxsize=2)
def _isolated_kernels(n: int, box: float):
    """FFTs of ``∇G = r/(4π r³)·cell³`` on the zero-padded ``(2n)³`` grid."""
    m = 2 * n
    cell = box / n
    off = np.fft.fftfreq(m, d=1.0 / m) * cell           # minimum-image offsets, 0 first
    x, y, z = off[:, None, None], off[None, :, None], off[None, None, :]
    r2 = x * x + y * y + z * z
    r2[0, 0, 0] = 1.0
    inv = cell ** 3 / (4.0 * np.pi * r2 ** 1.5)
    inv[0, 0, 0] = 0.0                                  # no self-force
    kernels = []
    for comp in (x, y, z):
        kk = np.fft.fftn(comp * inv)
        kk.flags.writeable = False
        kernels.append(kk)
    return tuple(kernels)


def pm_force_isolated(delta, box: float):
    """Open-boundary Poisson (zero-padded Hockney–Eastwood): ``∇φ`` of ``∇²φ = δ``, no images."""
    d = _cubic(delta)
    n, box = _check_grid(d.shape[0], box)
    m = 2 * n
    pad = np.zeros((m, m, m), dtype=np.float64)
    pad[:n, :n, :n] = d
    pad_k = np.fft.fftn(pad)
    return tuple(np.fft.ifftn(pad_k * kern).real[:n, :n, :n].copy()
                 for kern in _isolated_kernels(n, box))


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------

def zeldovich_pm_ic_from_field(cosmo: CosmologySpec, delta_z0, *, box: float,
                               z_start: float):
    """Growing-mode Zel'dovich particles (one per cell) from a z = 0 linear field.

    Returns ``(pos, vel)`` exactly as :func:`~oneuniverse.reference.linear.zeldovich`
    (Mpc/h in ``[0, box)``, km/s peculiar); ``vel`` is the growing mode
    ``p = a²·E·f·D·ψ`` in :func:`run_pm`'s momentum variable.
    """
    return zeldovich(delta_z0, cosmo, box=box, z=z_start)


def zeldovich_pm_ic(cosmo: CosmologySpec, *, box: float, n_grid: int, z_start: float,
                    seed: int):
    """``(pos, vel, ic_field)``: Zel'dovich ICs at ``z_start`` and the z = 0 linear δ they come from."""
    ic = gaussian_field(cosmo, box=box, n_grid=n_grid, z=0.0, seed=seed)
    pos, vel = zeldovich_pm_ic_from_field(cosmo, ic, box=box, z_start=z_start)
    return pos, vel, ic


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _time_integral(a0: float, a1: float, om: float, power: int) -> float:
    """``∫_{a0}^{a1} da / (a^power · E(a))``."""
    val, _ = integrate.quad(lambda a: 1.0 / (a ** power * _e(a, om)), a0, a1,
                            epsabs=0.0, epsrel=1e-10)
    return float(val)


def _delta(pos: np.ndarray, n: int, box: float) -> np.ndarray:
    rho = deposit_cic(pos, n, box)
    return rho / rho.mean() - 1.0


def _gradphi(pos: np.ndarray, n: int, box: float, force: str) -> np.ndarray:
    solve = pm_force if force == "periodic" else pm_force_isolated
    grids = solve(_delta(pos, n, box), box)
    return np.stack([interpolate_cic(g, pos, box) for g in grids], axis=1)


def run_pm(pos, vel, cosmo: CosmologySpec, *, box: float, n_grid: int, z_start: float,
           z_end: float, n_steps: int, force: str = "periodic",
           snapshots: Sequence[float] = ()) -> dict:
    """Evolve particles from ``z_start`` to ``z_end`` with a KDK particle-mesh.

    ``pos`` (Mpc/h, any value; wrapped) and ``vel`` (km/s peculiar) are
    ``(N, 3)``. Returns ``{"pos": (N, 3) in [0, box), "vel": (N, 3) km/s,
    "snapshots": {z: δ grid}}``. See the module docstring for the momentum
    variable, the time stepping and the snapshot handling.
    """
    _require_lcdm(cosmo)
    n, box = _check_grid(n_grid, box)
    x = _positions(pos)
    v = _positions(vel, "vel")
    if v.shape != x.shape:
        raise ValidationError(f"vel shape {v.shape} differs from pos shape {x.shape}")
    a_start, a_end = _scale_factor(z_start), _scale_factor(z_end)
    if z_end > z_start:
        raise ValidationError(f"z_end ({z_end}) must not exceed z_start ({z_start})")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValidationError(f"n_steps must be an integer >= 1, got {n_steps}")
    n_steps = int(n_steps)
    if force not in _FORCES:
        raise ValidationError(f"force must be one of {_FORCES}, got {force!r}")
    snap_a = {}
    for zs in snapshots:
        if not (z_end <= zs <= z_start):
            raise ValidationError(f"snapshot z={zs} outside [z_end={z_end}, z_start={z_start}]")
        snap_a[zs] = _scale_factor(zs)
    om = cosmo.omega_m

    x = np.mod(x, box)
    p = (a_start / 100.0) * v
    out_snaps = {zs: _delta(x, n, box) for zs, a_s in snap_a.items() if a_s <= a_start}
    pending = {zs: a_s for zs, a_s in snap_a.items() if zs not in out_snaps}

    a_grid = np.linspace(a_start, a_end, n_steps + 1)
    a_grid[-1] = a_end
    acc = -1.5 * om * _gradphi(x, n, box, force)
    for i in range(n_steps):
        a0, a1 = float(a_grid[i]), float(a_grid[i + 1])
        ah = 0.5 * (a0 + a1)
        p += acc * _time_integral(a0, ah, om, 2)
        for zs, a_s in list(pending.items()):
            if a_s < a1:
                xs = np.mod(x + p * _time_integral(a0, a_s, om, 3), box)
                out_snaps[zs] = _delta(xs, n, box)
                del pending[zs]
        x = np.mod(x + p * _time_integral(a0, a1, om, 3), box)
        acc = -1.5 * om * _gradphi(x, n, box, force)
        p += acc * _time_integral(ah, a1, om, 2)
    x[x >= box] -= box
    for zs in pending:          # a_s == a_end (to rounding)
        out_snaps[zs] = _delta(x, n, box)
    return {"pos": x, "vel": (100.0 / a_end) * p,
            "snapshots": {zs: out_snaps[zs] for zs in snap_a}}
