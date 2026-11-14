"""Linear theory of ``oneuniverse.reference.linear``: σ8, growth, fields, Zel'dovich."""
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.fieldstats import power_spectrum
from oneuniverse.reference.linear import (
    find_peaks,
    gaussian_field,
    growth_factor,
    growth_rate,
    k_grid,
    linear_power,
    zeldovich,
)
from oneuniverse.simulation.specs import CosmologySpec

from ._helpers import divergence_on_grid

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


def test_linear_power_sigma8_normalised():
    k = np.logspace(-4, 2, 2000); P = linear_power(k, COSMO); W = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x**3
    s8 = np.sqrt(np.trapezoid(k**2 * P * W(k * 8.0)**2, k) / (2 * np.pi**2)); assert s8 == pytest.approx(0.8111, rel=1e-3)


def test_growth_consistent_f_and_D():
    z = 0.5; eps = 1e-3; a = 1 / (1 + z); Dp = growth_factor(1 / (a * (1 + eps)) - 1, COSMO); Dm = growth_factor(1 / (a * (1 - eps)) - 1, COSMO)
    f_num = (np.log(Dp) - np.log(Dm)) / (2 * eps); assert growth_rate(z, COSMO) == pytest.approx(f_num, rel=1e-3) and growth_factor(0.0, COSMO) == 1.0


def test_gaussian_field_variance_matches_power():
    box, n = 500.0, 64; d = gaussian_field(COSMO, box=box, n_grid=n, seed=0); k, P = power_spectrum(d, box=box); kk = k[(k > 0.05) & (k < 0.3)]
    assert abs(d.mean()) < 1e-10 and np.allclose(P[(k > 0.05) & (k < 0.3)] / linear_power(kk, COSMO), 1.0, atol=0.25)


@pytest.mark.slow
def test_zeldovich_velocity_units_ensemble():
    box, n, z = 500.0, 64, 0.5; a = 1 / (1 + z); E = np.sqrt(COSMO.omega_m * a**-3 + 1 - COSMO.omega_m); f = growth_rate(z, COSMO); D = growth_factor(z, COSMO)
    ratios = []
    for seed in range(5):
        d0 = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); pos, vel = zeldovich(d0, COSMO, box=box, z=z)  # noqa: RUF059
        # divergence of the Lagrangian velocity field vs −a H f δ(z): compare power at k<0.1 using the displacement-field divergence computed on the grid
        div_v = divergence_on_grid(vel.reshape(n, n, n, 3), box); k, Pv = power_spectrum(div_v, box=box); k2, Pd = power_spectrum(-(100 * a * E * f) * D * d0, box=box)  # noqa: RUF059
        sel = k < 0.1; ratios.append(np.mean(Pv[sel] / Pd[sel]))
    assert abs(np.mean(ratios) - 1.0) < 0.02


# --- supplementary unit tests (fast) -------------------------------------


def test_growth_rejects_non_lcdm():
    with pytest.raises(ValidationError):
        growth_factor(0.5, CosmologySpec(0.7, 0.3, 0.05, 0.8, 0.96, w0=-0.9))
    with pytest.raises(ValidationError):
        growth_rate(0.5, CosmologySpec(0.7, 0.3, 0.05, 0.8, 0.96, wa=0.1))


def test_growth_matter_dominated_limit():
    # Einstein–de Sitter-like at high z: D ∝ a, f → 1.
    assert growth_rate(200.0, COSMO) == pytest.approx(1.0, abs=1e-4)
    r = growth_factor(200.0, COSMO) / growth_factor(100.0, COSMO)
    assert r == pytest.approx(101.0 / 201.0, rel=1e-4)


def test_linear_power_scales_with_growth_squared():
    k = np.array([0.01, 0.1, 1.0])
    D = growth_factor(1.0, COSMO)
    assert np.allclose(linear_power(k, COSMO, z=1.0), D**2 * linear_power(k, COSMO), rtol=1e-12)


def test_k_grid_full_fft_layout():
    kx, _, _, kmag = k_grid(8, 80.0)
    assert kmag.shape == (8, 8, 8)
    kf = 2 * np.pi / 80.0
    assert np.allclose(np.ravel(kx), kf * np.fft.fftfreq(8, d=1 / 8))
    assert kmag[0, 0, 0] == 0.0 and kmag[1, 2, 3] == pytest.approx(kf * np.sqrt(14))


def test_gaussian_field_real_seeded_and_mode_sum_variance():
    box, n = 200.0, 32
    d = gaussian_field(COSMO, box=box, n_grid=n, seed=3)
    assert d.dtype == np.float64 and d.shape == (n, n, n)
    assert np.array_equal(d, gaussian_field(COSMO, box=box, n_grid=n, seed=3))
    _, _, _, kmag = k_grid(n, box)
    pk = np.zeros_like(kmag)
    pk[kmag > 0] = linear_power(kmag[kmag > 0], COSMO)
    expected = pk.sum() / box**3
    assert d.var() == pytest.approx(expected, rel=0.1)


def test_zeldovich_positions_and_velocity_formula():
    box, n, z = 200.0, 16, 1.0
    d0 = gaussian_field(COSMO, box=box, n_grid=n, seed=1)
    _, vel0 = zeldovich(d0, COSMO, box=box, z=0.0)
    pos, vel = zeldovich(d0, COSMO, box=box, z=z)
    assert pos.shape == vel.shape == (n**3, 3)
    assert np.all((pos >= 0) & (pos < box))
    a = 0.5
    E = np.sqrt(COSMO.omega_m * a**-3 + 1 - COSMO.omega_m)
    D, f = growth_factor(z, COSMO), growth_rate(z, COSMO)
    f0 = growth_rate(0.0, COSMO)
    # velocity is linear in ψ: v(z) / v(0) = a E(a) f(z) D(z) / f(0)
    assert np.allclose(vel, vel0 * (a * E * f * D / f0), rtol=1e-10, atol=1e-10)
    # the displacement divergence reproduces −D δ on every mode off the Nyquist
    # planes (where the axis derivative is zeroed by convention)
    psi = vel.reshape(n, n, n, 3) / (100 * a * E * f)
    lhs, rhs = np.fft.fftn(divergence_on_grid(psi, box)), np.fft.fftn(-D * d0)
    off = np.ones((n, n, n), dtype=bool)
    off[n // 2, :, :] = off[:, n // 2, :] = off[:, :, n // 2] = False
    assert np.allclose(lhs[off], rhs[off], rtol=0, atol=1e-9 * np.abs(rhs).max())


def test_find_peaks_single_spike():
    n, box = 16, 32.0
    d = np.zeros((n, n, n))
    d[3, 4, 5] = 10.0
    d[3, 4, 6] = 2.0      # positive neighbour: inside the 3³ neighbourhood
    d[0, 0, 0] = -1.0
    t = find_peaks(d, box=box, threshold=2.0)
    assert t.column_names == ["x", "y", "z", "height", "mass_proxy"]
    assert t.num_rows == 1
    row = t.to_pylist()[0]
    cell = box / n
    assert (row["x"], row["y"], row["z"]) == (3.5 * cell, 4.5 * cell, 5.5 * cell)
    assert row["height"] == pytest.approx(10.0 / d.std())
    assert row["mass_proxy"] == pytest.approx(12.0 * cell**3)


def test_find_peaks_periodic_and_empty():
    n, box = 8, 8.0
    d = np.zeros((n, n, n))
    d[0, 0, 0] = 5.0
    d[n - 1, n - 1, n - 1] = 6.0    # periodic neighbour of (0,0,0) and higher
    t = find_peaks(d, box=box, threshold=1.0)
    assert t.num_rows == 1 and t.column("x")[0].as_py() == 7.5
    assert find_peaks(np.zeros((n, n, n)) + 1e-3 * np.arange(n)[:, None, None], box=box, threshold=100.0).num_rows == 0
