"""Mask-aware Wiener filter, constrained realisation and lognormal mocks.

``oneuniverse.reference.reconstruct`` and ``oneuniverse.reference.mock``.
"""
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.fieldstats import cross_power, cross_r, power_spectrum
from oneuniverse.reference.linear import gaussian_field
from oneuniverse.reference.mock import (
    lognormal_tracers,
    mask_ball,
    mask_slab,
    run_mock_challenge,
)
from oneuniverse.reference.reconstruct import (
    constrained_realization,
    wiener_reconstruct,
)
from oneuniverse.simulation.specs import CosmologySpec

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


# ---- brief tests (verbatim) ------------------------------------------------

def test_constrained_requires_seed():
    with pytest.raises(TypeError): constrained_realization(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3, bias=1.0, z=0.0)

@pytest.mark.slow
def test_constrained_realization_independent_of_truth_seed():
    box, n = 256.0, 64; hi, lo = [], []
    for seed in range(5):
        truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); dg = lognormal_tracers(truth, box=box, nbar=5e-3, bias=1.5, seed=100 + seed); dg = dg / dg.mean() - 1
        wf = wiener_reconstruct(dg, COSMO, box=box, nbar=5e-3, bias=1.5); cr = constrained_realization(dg, COSMO, box=box, nbar=5e-3, bias=1.5, z=0.0, seed=seed + 1000)
        k, r_t = cross_r(cr, truth, box=box); _, r_w = cross_r(cr, wf, box=box); knyq = np.pi * n / box
        hi.append(np.mean(r_t[k > 0.5 * knyq])); lo.append(np.mean(r_w[k < 0.1]))
    assert np.mean(hi) < 0.5 and np.mean(lo) > 0.95

@pytest.mark.slow
def test_masked_wiener_recovers_inside_footprint():
    box, n = 256.0, 64; diffs = []
    for seed in range(5):
        truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); dg = lognormal_tracers(truth, box=box, nbar=5e-3, bias=1.5, seed=200 + seed); dg = dg / dg.mean() - 1
        mask = mask_slab(n, 0, 0.0, 0.5); wf_m = wiener_reconstruct(dg * mask, COSMO, box=box, nbar=5e-3, bias=1.5, mask=mask); wf = wiener_reconstruct(dg, COSMO, box=box, nbar=5e-3, bias=1.5)
        k, r_m = cross_r(wf_m * mask, truth * mask, box=box); _, r = cross_r(wf * mask, truth * mask, box=box); sel = k < 0.2; diffs.append(np.mean(np.abs(r_m[sel] - r[sel])))
    assert np.mean(diffs) < 0.05

def test_mock_challenge_returns_metrics():
    out = run_mock_challenge(COSMO, box=128.0, n_grid=32, nbar=5e-3, seed=0); assert set(out) == {"truth", "delta_g", "wiener", "metrics"} and 0 < out["metrics"].k_half


# ---- supplementary tests ---------------------------------------------------

def _dg(truth, *, box, nbar, bias, seed):
    counts = lognormal_tracers(truth, box=box, nbar=nbar, bias=bias, seed=seed)
    return counts / counts.mean() - 1.0


def test_lognormal_tracers_counts_and_mean():
    box, n, nbar = 256.0, 32, 5e-3
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=3)
    counts = lognormal_tracers(truth, box=box, nbar=nbar, bias=1.5, seed=7)
    assert counts.shape == (n, n, n) and counts.dtype == np.float64
    assert np.all(counts >= 0) and np.all(counts == np.round(counts))
    expected = nbar * box ** 3
    assert abs(counts.sum() - expected) < 5 * np.sqrt(expected) + 0.05 * expected
    again = lognormal_tracers(truth, box=box, nbar=nbar, bias=1.5, seed=7)
    assert np.array_equal(counts, again)
    assert not np.array_equal(counts, lognormal_tracers(truth, box=box, nbar=nbar, bias=1.5, seed=8))


@pytest.mark.parametrize("bias", [1.0, 1.5, 2.0])
def test_lognormal_tracers_cross_bias_within_5_percent(bias):
    # sigma_cell of the matter field below 1 (box 512, 16 Mpc/h cells); 16
    # seeds: the non-Gaussian scatter of the low-k cross bias is ~5 %/seed at b = 2.
    box, n, nbar = 512.0, 32, 2e-2
    ratios = []
    for seed in range(16):
        truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed)
        assert truth.std() < 1.0
        dg = _dg(truth, box=box, nbar=nbar, bias=bias, seed=50 + seed)
        k, pc = cross_power(dg, truth, box=box)
        _, pt = power_spectrum(truth, box=box)
        sel = k < 0.1
        ratios.append(pc[sel].sum() / pt[sel].sum())
    assert abs(np.mean(ratios) / bias - 1.0) < 0.05


def test_wiener_noise_free_limit_returns_matter_field():
    box, n, bias = 200.0, 32, 2.0
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=1)
    wf = wiener_reconstruct(bias * truth, COSMO, box=box, nbar=1e9, bias=bias)
    assert np.allclose(wf, truth, atol=1e-4 * truth.std())
    assert abs(wf.mean()) < 1e-12


def test_wiener_gain_is_signal_over_signal_plus_noise():
    # A single Fourier mode is filtered by S/(S+N)/b with S = b^2 P_lin(k, z).
    from oneuniverse.reference.linear import linear_power
    box, n, bias, nbar, z = 100.0, 16, 1.3, 1e-3, 0.5
    x = (np.arange(n) + 0.5) * box / n
    kf = 2 * np.pi / box
    d = np.cos(3 * kf * x)[:, None, None] * np.ones((1, n, n))
    s = bias ** 2 * linear_power(np.array([3 * kf]), COSMO, z=z)[0]
    expect = s / (s + 1.0 / nbar) / bias * d
    wf = wiener_reconstruct(d, COSMO, box=box, nbar=nbar, bias=bias, z=z)
    assert np.allclose(wf, expect, atol=1e-12)


def test_wiener_mask_none_equals_full_mask():
    box, n = 128.0, 16
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=2)
    dg = _dg(truth, box=box, nbar=1e-2, bias=1.5, seed=3)
    full = np.ones((n, n, n), dtype=bool)
    a = wiener_reconstruct(dg, COSMO, box=box, nbar=1e-2, bias=1.5)
    b = wiener_reconstruct(dg, COSMO, box=box, nbar=1e-2, bias=1.5, mask=full)
    assert np.allclose(a, b)


def test_wiener_mask_renormalises_inside_and_ignores_outside():
    box, n = 128.0, 16
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=4)
    dg = _dg(truth, box=box, nbar=1e-2, bias=1.5, seed=5)
    mask = mask_slab(n, 2, 0.25, 0.75)
    junk = np.where(mask, dg, 1e3)  # values outside the footprint are ignored
    a = wiener_reconstruct(dg, COSMO, box=box, nbar=1e-2, bias=1.5, mask=mask)
    b = wiener_reconstruct(junk, COSMO, box=box, nbar=1e-2, bias=1.5, mask=mask)
    assert np.allclose(a, b)
    # a constant offset of the density inside the footprint is absorbed by
    # the renormalisation (1 + d) / <1 + d>_mask - 1
    c = wiener_reconstruct(np.where(mask, 1.2 * (1 + dg) - 1, 0.0), COSMO, box=box,
                           nbar=1e-2, bias=1.5, mask=mask)
    assert np.allclose(a, c)
    # far outside the footprint the estimate relaxes to the prior mean 0
    assert np.abs(a[:, :, 0]).mean() < 0.2 * np.abs(a[:, :, n // 2]).mean()


def test_constrained_realization_deterministic_and_seed_dependent():
    box, n = 128.0, 16
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=0)
    dg = _dg(truth, box=box, nbar=5e-3, bias=1.5, seed=1)
    kw = {"box": box, "nbar": 5e-3, "bias": 1.5, "z": 0.0}
    a = constrained_realization(dg, COSMO, seed=11, **kw)
    assert np.array_equal(a, constrained_realization(dg, COSMO, seed=11, **kw))
    assert not np.allclose(a, constrained_realization(dg, COSMO, seed=12, **kw))


def test_constrained_realization_does_not_reproduce_truth_with_same_seed():
    # Even when the caller reuses the truth's integer seed, the random signal
    # of the CR comes from a derived stream, so its small scales are not the
    # truth's (the old engine's defect).
    box, n, seed = 256.0, 32, 4
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed)
    dg = _dg(truth, box=box, nbar=5e-3, bias=1.5, seed=99)
    kw = {"box": box, "nbar": 5e-3, "bias": 1.5, "z": 0.0}
    cr = constrained_realization(dg, COSMO, seed=seed, **kw)
    wf = wiener_reconstruct(dg, COSMO, box=box, nbar=5e-3, bias=1.5)
    k, r = cross_r(cr - wf, truth, box=box)
    # reusing the truth's white noise would give r ~ 0.2-0.4 here
    assert np.mean(np.abs(r[k > 0.5 * np.pi * n / box])) < 0.1


def test_constrained_realization_has_prior_power():
    # Gaussian data matching the filter's model (b·δ + white noise of power
    # 1/nbar); lognormal data would carry extra auto power at high k.
    box, n, nbar = 256.0, 32, 2e-4
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=6)
    noise = np.random.default_rng(7).standard_normal((n, n, n)) / np.sqrt(nbar * (box / n) ** 3)
    dg = 1.5 * truth + noise
    cr = constrained_realization(dg, COSMO, box=box, nbar=nbar, bias=1.5, z=0.0, seed=8)
    wf = wiener_reconstruct(dg, COSMO, box=box, nbar=nbar, bias=1.5)
    k, p_cr = power_spectrum(cr, box=box)
    _, p_wf = power_spectrum(wf, box=box)
    _, p_t = power_spectrum(truth, box=box)
    hi = k > 0.5 * np.pi * n / box
    # the WF is power-suppressed at high k; the CR restores the prior power
    assert p_wf[hi].sum() < 0.8 * p_cr[hi].sum()
    assert 0.6 < p_cr[hi].sum() / p_t[hi].sum() < 1.6


def test_constrained_realization_masked_runs():
    box, n = 128.0, 16
    truth = gaussian_field(COSMO, box=box, n_grid=n, seed=0)
    mask = mask_ball(n, (0.5, 0.5, 0.5), 0.3)
    dg = np.where(mask, _dg(truth, box=box, nbar=5e-3, bias=1.5, seed=1), 0.0)
    cr = constrained_realization(dg, COSMO, box=box, nbar=5e-3, bias=1.5, z=0.0, seed=3, mask=mask)
    assert cr.shape == (n, n, n) and np.all(np.isfinite(cr))


def test_mask_slab_geometry():
    m = mask_slab(8, 1, 0.25, 0.5)
    assert m.dtype == bool and m.shape == (8, 8, 8)
    # centres (i + 0.5)/8 in [0.25, 0.5): i = 2, 3
    assert np.array_equal(np.flatnonzero(m[0, :, 0]), [2, 3])
    assert m[:, 2, :].all() and not m[:, 4, :].any()
    assert mask_slab(8, 0, 0.0, 1.0).all()


def test_mask_ball_geometry():
    n = 16
    m = mask_ball(n, (0.5, 0.5, 0.5), 0.25)
    c = (np.arange(n) + 0.5) / n
    r = np.sqrt((c[:, None, None] - 0.5) ** 2 + (c[None, :, None] - 0.5) ** 2 + (c[None, None, :] - 0.5) ** 2)
    assert m.dtype == bool and np.array_equal(m, r <= 0.25)
    # non-periodic: a ball centred on a corner does not wrap to the far side
    corner = mask_ball(n, (0.0, 0.0, 0.0), 0.2)
    assert corner[0, 0, 0] and not corner[-1, -1, -1] and not corner[0, 0, -1]


@pytest.mark.parametrize("call", [
    lambda: wiener_reconstruct(np.zeros((8, 8, 4)), COSMO, box=100.0, nbar=1e-3),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=0.0),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3, bias=0.0),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=-1.0, nbar=1e-3),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3,
                               mask=np.ones((4, 4, 4), bool)),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3,
                               mask=np.zeros((8, 8, 8), bool)),
    lambda: wiener_reconstruct(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3,
                               mask=np.full((8, 8, 8), 0.5)),
    lambda: wiener_reconstruct(np.full((8, 8, 8), -1.0), COSMO, box=100.0, nbar=1e-3),
    lambda: constrained_realization(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3,
                                    bias=1.0, z=0.0, seed=None),
    lambda: constrained_realization(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3,
                                    bias=1.0, z=0.0, seed=1.5),
    lambda: lognormal_tracers(np.zeros((8, 8, 8)), box=100.0, nbar=-1.0, bias=1.0, seed=0),
    lambda: lognormal_tracers(np.zeros((8, 8, 8)), box=100.0, nbar=1e-3, bias=0.0, seed=0),
    lambda: lognormal_tracers(np.zeros((8, 8)), box=100.0, nbar=1e-3, bias=1.0, seed=0),
    lambda: mask_slab(8, 3, 0.0, 0.5),
    lambda: mask_slab(8, 0, 0.5, 0.5),
    lambda: mask_slab(8, 0, -0.1, 0.5),
    lambda: mask_ball(8, (0.5, 0.5), 0.2),
    lambda: mask_ball(8, (0.5, 0.5, 0.5), 0.0),
    lambda: run_mock_challenge(COSMO, box=128.0, n_grid=16, nbar=0.0, seed=0),
])
def test_bad_arguments_raise_validation_error(call):
    with pytest.raises(ValidationError):
        call()
