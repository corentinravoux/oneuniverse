"""Eisenstein & Hu (1998) no-wiggle linear power spectrum, pure numpy.

Reference: Eisenstein & Hu 1998, ApJ 496, 605 (arXiv:astro-ph/9709112),
the "no-wiggle" shape fit (eqs. 26-31). Wavenumbers ``k`` in h/Mpc;
P(k) in (Mpc/h)^3. Normalised so that sigma8 reproduces the input.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def transfer_eh_nowiggle(k, cosmo: CosmologySpec) -> np.ndarray:
    """No-wiggle EH98 transfer function T(k). ``k`` in h/Mpc."""
    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    om = c.omega_m
    ob = c.omega_b
    h = c.h
    theta = c.t_cmb / 2.7
    omh2 = om * h * h
    obh2 = ob * h * h
    f_baryon = ob / om
    # sound horizon in Mpc/h (the trailing * h converts Mpc -> Mpc/h)
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2 ** 0.75) * h
    alpha = (
        1.0
        - 0.328 * np.log(431.0 * omh2) * f_baryon
        + 0.38 * np.log(22.3 * omh2) * f_baryon ** 2
    )
    # effective shape; k in h/Mpc, s in Mpc/h -> k*s dimensionless
    gamma_eff = om * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s) ** 4))
    q = k * theta ** 2 / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q * q)


def unnormalised_power(k, cosmo: CosmologySpec) -> np.ndarray:
    """Shape-only P(k) ~ k^n_s T(k)^2 (arbitrary amplitude)."""
    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    t = transfer_eh_nowiggle(k, c)
    return k ** c.n_s * t ** 2


def _top_hat(x: np.ndarray) -> np.ndarray:
    """Fourier top-hat window W(x); W(0) = 1."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x > 1e-8
    xn = x[nz]
    out[nz] = 3.0 * (np.sin(xn) - xn * np.cos(xn)) / xn ** 3
    return out


def sigma_R(R: float, cosmo: CosmologySpec, *, pk_func=None) -> float:
    """RMS density fluctuation in a top-hat of radius ``R`` (Mpc/h).

    If ``pk_func`` is None, integrates the *unnormalised* shape (used to
    derive the normalisation); otherwise integrates ``pk_func(k)``.
    """
    c = require_cosmo(cosmo)
    k = np.logspace(-4.0, 2.0, 4000)
    pk = unnormalised_power(k, c) if pk_func is None else np.asarray(pk_func(k))
    w = _top_hat(k * R)
    integrand = k ** 2 * pk * w ** 2
    sigma2 = np.trapz(integrand, k) / (2.0 * np.pi ** 2)
    return float(np.sqrt(sigma2))


def normalisation(cosmo: CosmologySpec) -> float:
    """Amplitude A such that sigma8 of A * unnormalised_power == sigma8."""
    c = require_cosmo(cosmo)
    s8_unnorm = sigma_R(8.0, c, pk_func=None)
    return float((c.sigma8 / s8_unnorm) ** 2)


def linear_power(k, cosmo: CosmologySpec, z: float = 0.0) -> np.ndarray:
    """σ8-normalised linear matter P(k) at redshift ``z`` (Mpc/h)^3."""
    from oneuniverse.simulation.linear.growth import growth_factor

    c = require_cosmo(cosmo)
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    amp = normalisation(c)
    d = growth_factor(z, c)
    return amp * unnormalised_power(k, c) * d * d
