"""Particle-mesh of ``oneuniverse.reference.pm``: CIC, Poisson force, KDK driver."""
import numpy as np
import pytest
from fixtures.cosmology import COSMO

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.fieldstats import cross_r, power_spectrum
from oneuniverse.reference.linear import (
    gaussian_field,
    growth_factor,
    zeldovich,
)
from oneuniverse.reference.pm import (
    deposit_cic,
    interpolate_cic,
    pm_force,
    pm_force_isolated,
    run_pm,
    zeldovich_pm_ic,
    zeldovich_pm_ic_from_field,
)

# ---- brief tests (verbatim) ------------------------------------------------

def test_cic_deposit_conserves_mass_and_is_centred():
    pos = np.array([[0.5 * (100 / 16)] * 3]); g = deposit_cic(pos, 16, 100.0); assert g.sum() == pytest.approx(1.0) and g[0, 0, 0] == pytest.approx(1.0)   # a particle at the centre of cell 0 lands entirely in cell 0
    pos2 = np.random.default_rng(0).uniform(0, 100, (10_000, 3)); assert deposit_cic(pos2, 16, 100.0).sum() == pytest.approx(10_000)


def test_pm_force_is_gradient_of_potential_for_single_mode():
    n, box = 32, 100.0; x = (np.arange(n) + 0.5) * box / n; d = np.sin(2 * np.pi * x / box)[:, None, None] * np.ones((1, n, n)); fx, fy, fz = pm_force(d, box)
    kf = 2 * np.pi / box; assert np.allclose(fx, np.cos(2 * np.pi * x / box)[:, None, None] / kf, rtol=1e-6) and np.allclose(fy, 0) and np.allclose(fz, 0)


@pytest.mark.slow
def test_pm_linear_growth_ensemble():
    box, n = 256.0, 64; ratios, rs = [], []
    for seed in range(5):
        pos, vel, ic = zeldovich_pm_ic(COSMO, box=box, n_grid=n, z_start=9.0, seed=seed); out = run_pm(pos, vel, COSMO, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=20)
        d = deposit_cic(out["pos"], n, box); d = d / d.mean() - 1; k, P = power_spectrum(d, box=box); sel = k < 0.05; _, P_ic = power_spectrum(ic, box=box)
        # vs the same realisation grown linearly, not P_lin: cosmic variance (one k bin of 9 modes below 0.05 at box 256); owner decision 2026-11-14, design §9b deviation
        ratios.append(np.mean(P[sel] / (growth_factor(0.0, COSMO) ** 2 * P_ic[sel]))); rs.append(np.mean(cross_r(d, ic, box=box)[1][sel]))
    assert abs(np.mean(ratios) - 1.0) < 0.05 and np.mean(rs) > 0.98


# ---- CIC -------------------------------------------------------------------

def test_cic_split_wraps_periodically_and_accepts_any_position():
    n, box = 8, 80.0
    # on the boundary between cell 7 and (wrapped) cell 0 along x
    g = deposit_cic(np.array([[0.0, 5.0, 5.0]]), n, box)
    assert g[7, 0, 0] == pytest.approx(0.5) and g[0, 0, 0] == pytest.approx(0.5)
    # positions outside [0, box) are wrapped first
    p = np.random.default_rng(1).uniform(0, box, (500, 3))
    assert np.allclose(deposit_cic(p + 3 * box, n, box), deposit_cic(p, n, box))
    assert np.allclose(deposit_cic(p - box, n, box), deposit_cic(p, n, box))


def test_interpolate_cic_is_adjoint_of_deposit():
    n, box = 12, 50.0
    rng = np.random.default_rng(2)
    pos = rng.uniform(-box, 2 * box, (300, 3))
    grid = rng.standard_normal((n, n, n))
    w = rng.standard_normal(300)
    lhs = np.sum(interpolate_cic(grid, pos, box) * w)
    # deposit with weights == sum_i w_i * (CIC kernel of particle i)
    dep = sum(wi * deposit_cic(pos[i:i + 1], n, box) for i, wi in enumerate(w))
    assert lhs == pytest.approx(np.sum(grid * dep), rel=1e-12)
    # interpolating at a cell centre returns that cell's value
    centre = np.array([[(3 + 0.5) * box / n, (4 + 0.5) * box / n, (5 + 0.5) * box / n]])
    assert interpolate_cic(grid, centre, box)[0] == pytest.approx(grid[3, 4, 5])


def test_cic_rejects_bad_input():
    with pytest.raises(ValidationError):
        deposit_cic(np.zeros((4, 2)), 8, 10.0)
    with pytest.raises(ValidationError):
        deposit_cic(np.zeros((4, 3)), 8, -1.0)
    with pytest.raises(ValidationError):
        interpolate_cic(np.zeros((4, 4, 5)), np.zeros((2, 3)), 1.0)


# ---- Poisson ---------------------------------------------------------------

def test_isolated_force_of_point_mass_has_no_periodic_images():
    n, box = 32, 64.0
    cell = box / n
    d = np.zeros((n, n, n)); d[2, 16, 16] = 1.0
    fxp, _, _ = pm_force(d, box)
    fxi, fyi, fzi = pm_force_isolated(d, box)
    probe = (n - 3, 16, 16)                       # far side of the box along x
    r = (n - 3 - 2) * cell
    expected = cell ** 3 / (4 * np.pi * r ** 2)   # |∇φ| of ∇²φ = δ for the point "mass" δ·cell³
    # the force F = −∇φ is attractive: at the probe (x > mass) it points to −x, toward the mass
    assert fxi[probe] == pytest.approx(-expected, rel=0.02)
    assert abs(fyi[probe]) < 1e-3 * expected and abs(fzi[probe]) < 1e-3 * expected
    # periodic solve: the nearest image (across the x boundary) dominates, sign flips
    assert fxp[probe] > 0
    # 1/r² fall-off: 4 cells from the mass vs 27 cells
    assert fxi[6, 16, 16] == pytest.approx(fxi[probe] * (27 / 4) ** 2, rel=1e-6)
    # and on the other side of the mass it points the other way (+x, still toward the mass)
    assert fxi[0, 16, 16] > 0


def test_isolated_force_matches_periodic_for_compact_mode_in_large_box_centre():
    # a compact zero-mean blob far from the edges: isolated and periodic agree at the centre
    n, box = 32, 64.0
    x = (np.arange(n) + 0.5) * box / n - box / 2
    r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    d = np.exp(-r2 / (2 * 3.0 ** 2)) - (3.0 / 1.6) ** 3 * np.exp(-r2 / (2 * 4.8 ** 2))  # net-zero mass
    fp = pm_force(d, box)[0]; fi = pm_force_isolated(d, box)[0]
    c = slice(12, 20)
    assert np.allclose(fi[c, c, c], fp[c, c, c], atol=0.05 * np.abs(fp[c, c, c]).max())


def test_pm_force_rejects_non_cubic():
    with pytest.raises(ValidationError):
        pm_force(np.zeros((4, 4, 5)), 1.0)
    with pytest.raises(ValidationError):
        pm_force_isolated(np.zeros((4, 4)), 1.0)


# ---- initial conditions ----------------------------------------------------

def test_zeldovich_pm_ic_matches_linear_zeldovich():
    box, n, z = 200.0, 16, 9.0
    pos, vel, ic = zeldovich_pm_ic(COSMO, box=box, n_grid=n, z_start=z, seed=3)
    assert np.array_equal(ic, gaussian_field(COSMO, box=box, n_grid=n, z=0.0, seed=3))
    p2, v2 = zeldovich(ic, COSMO, box=box, z=z)
    assert np.allclose(pos, p2) and np.allclose(vel, v2)
    p3, v3 = zeldovich_pm_ic_from_field(COSMO, ic, box=box, z_start=z)
    assert np.array_equal(pos, p3) and np.array_equal(vel, v3)
    assert pos.shape == (n ** 3, 3) and pos.dtype == np.float64


# ---- driver ----------------------------------------------------------------

def _plane_wave_run(n, box=256.0, z_end=1.0):
    x = (np.arange(n) + 0.5) * box / n
    ic = 1e-3 * np.cos(2 * np.pi * x / box)[:, None, None] * np.ones((1, n, n))
    pos, vel = zeldovich_pm_ic_from_field(COSMO, ic, box=box, z_start=9.0)
    out = run_pm(pos, vel, COSMO, box=box, n_grid=n, z_start=9.0, z_end=z_end, n_steps=10,
                 snapshots=(9.0,))
    amp = lambda d: np.abs(np.fft.fftn(d)[1, 0, 0])
    d = deposit_cic(out["pos"], n, box); d = d / d.mean() - 1
    growth = amp(d) / amp(out["snapshots"][9.0]) / (growth_factor(z_end, COSMO) / growth_factor(9.0, COSMO))
    _, v_lin = zeldovich(ic, COSMO, box=box, z=z_end)
    vel_slope = np.sum(out["vel"][:, 0] * v_lin[:, 0]) / np.sum(v_lin[:, 0] ** 2)
    return growth, vel_slope


def test_run_pm_plane_wave_follows_linear_growth_and_velocity_and_converges():
    # linear plane wave (k = k_f): δ grows as D(a), vel matches 100·a·E·f·D·ψ (km/s);
    # the CIC/mesh deficit (~(k·cell)²) halves at least when the mesh is refined
    g32, v32 = _plane_wave_run(32)
    g16, v16 = _plane_wave_run(16)
    assert abs(g32 - 1) < 0.025 and abs(v32 - 1) < 0.025
    assert abs(g32 - 1) < 0.5 * abs(g16 - 1) and abs(v32 - 1) < 0.5 * abs(v16 - 1)


def test_run_pm_tracks_its_own_ic_amplitude_and_correlation():
    # random field, coarse mesh: cross-correlation with the IC stays ~1 at the fundamental
    box, n = 256.0, 32
    pos, vel, ic = zeldovich_pm_ic(COSMO, box=box, n_grid=n, z_start=9.0, seed=0)
    out = run_pm(pos, vel, COSMO, box=box, n_grid=n, z_start=9.0, z_end=1.0, n_steps=10)
    d = deposit_cic(out["pos"], n, box); d = d / d.mean() - 1
    k, P = power_spectrum(d, box=box); _, Pic = power_spectrum(ic, box=box)
    sel = k < 0.04
    ratio = np.mean(P[sel] / (growth_factor(1.0, COSMO) ** 2 * Pic[sel]))
    assert abs(ratio - 1) < 0.1 and np.mean(cross_r(d, ic, box=box)[1][sel]) > 0.99


def test_run_pm_snapshots_and_output_units():
    box, n = 200.0, 16
    pos, vel, _ = zeldovich_pm_ic(COSMO, box=box, n_grid=n, z_start=9.0, seed=1)
    out = run_pm(pos, vel, COSMO, box=box, n_grid=n, z_start=9.0, z_end=2.0, n_steps=4,
                 snapshots=(9.0, 3.3, 2.0))
    assert set(out) == {"pos", "vel", "snapshots"} and set(out["snapshots"]) == {9.0, 3.3, 2.0}
    assert out["pos"].shape == pos.shape and np.all((out["pos"] >= 0) & (out["pos"] < box))
    d_end = deposit_cic(out["pos"], n, box); d_end = d_end / d_end.mean() - 1
    assert np.allclose(out["snapshots"][2.0], d_end)
    d0 = deposit_cic(pos, n, box); assert np.allclose(out["snapshots"][9.0], d0 / d0.mean() - 1)
    # intermediate snapshot amplitude sits between the two ends
    s = [out["snapshots"][z].std() for z in (9.0, 3.3, 2.0)]
    assert s[0] < s[1] < s[2]
    # still near-linear at z=2: the velocity field is the ZA one (km/s, same units)
    _, v2 = zeldovich(gaussian_field(COSMO, box=box, n_grid=n, seed=1), COSMO, box=box, z=2.0)
    assert np.corrcoef(out["vel"][:, 0], v2[:, 0])[0, 1] > 0.95


def test_run_pm_zero_steps_forbidden_and_other_validation():
    pos, vel, _ = zeldovich_pm_ic(COSMO, box=100.0, n_grid=8, z_start=9.0, seed=0)
    kw = {"box": 100.0, "n_grid": 8, "z_start": 9.0, "z_end": 0.0}
    with pytest.raises(ValidationError):
        run_pm(pos, vel, COSMO, **{**kw, "z_end": 10.0}, n_steps=2)
    with pytest.raises(ValidationError):
        run_pm(pos, vel, COSMO, **kw, n_steps=0)
    with pytest.raises(ValidationError):
        run_pm(pos, vel, COSMO, **kw, n_steps=2, force="tree")
    with pytest.raises(ValidationError):
        run_pm(pos, vel, COSMO, **kw, n_steps=2, snapshots=(12.0,))
    with pytest.raises(ValidationError):
        run_pm(pos, vel[:-1], COSMO, **kw, n_steps=2)


def test_run_pm_isolated_runs_and_conserves_particles():
    pos, vel, _ = zeldovich_pm_ic(COSMO, box=100.0, n_grid=8, z_start=9.0, seed=0)
    out = run_pm(pos, vel, COSMO, box=100.0, n_grid=8, z_start=9.0, z_end=3.0, n_steps=3,
                 force="isolated")
    assert out["pos"].shape == pos.shape and np.all(np.isfinite(out["vel"]))
