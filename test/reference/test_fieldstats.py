"""``oneuniverse.reference.fieldstats``: the single P(k) / r(k) implementation."""
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.fieldstats import (
    RecoveryMetrics,
    bin_edges_kf,
    cell_corr,
    cross_power,
    cross_r,
    k_half,
    power_spectrum,
    recover_metrics,
    transfer,
)


def test_fieldstats_binning_and_r_of_identical_fields():
    d = np.random.default_rng(0).normal(size=(32, 32, 32)); k, r = cross_r(d, d, box=100.0); assert np.allclose(r, 1.0) and k[0] > 0 and np.all(np.diff(k) > 0)
    assert k_half(np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.5, 0.1])) == pytest.approx(0.2)
    m = recover_metrics(d, d, box=100.0); assert np.allclose(m.transfer, 1.0) and np.allclose(m.power_ratio, 1.0)


# --- supplementary unit tests ---------------------------------------------


def test_bin_edges_fundamental_spacing():
    n, box = 16, 100.0
    kf = 2 * np.pi / box
    e = bin_edges_kf(n, box)
    assert e[0] == pytest.approx(kf / 2)
    assert np.allclose(np.diff(e), kf)
    assert e[-1] <= np.pi * n / box + 1e-12
    assert e[-1] + kf > np.pi * n / box


def test_white_noise_power_level():
    # white noise of unit variance: P = V_cell = (box/n)³ per mode
    n, box = 32, 64.0
    d = np.random.default_rng(1).normal(size=(n, n, n))
    _, P = power_spectrum(d, box=box)
    assert np.mean(P) == pytest.approx((box / n) ** 3, rel=0.03)


def test_single_mode_power():
    n, box = 16, 50.0
    x = (np.arange(n) + 0.5) * box / n
    d = np.cos(2 * np.pi * 3 * x / box)[:, None, None] * np.ones((1, n, n))
    k, P = power_spectrum(d, box=box)
    kf = 2 * np.pi / box
    j = np.argmin(np.abs(k - 3 * kf))
    assert abs(k[j] - 3 * kf) < kf / 2   # mean |k| of the shell's modes
    assert P.sum() == pytest.approx(P[j])
    # |δ_k|² = (N/2)² for the two modes ±k; P = |δ_k|² V / N² = V/4 each,
    # averaged over every grid mode of the shell 2.5 kf <= |k| < 3.5 kf
    m = np.fft.fftfreq(n, d=1 / n)
    mm = np.sqrt(m[:, None, None] ** 2 + m[None, :, None] ** 2 + m[None, None, :] ** 2)
    n_modes = np.count_nonzero((mm >= 2.5) & (mm < 3.5))
    assert P[j] == pytest.approx(2 * (box**3 / 4) / n_modes, rel=1e-12)


def test_cross_quantities_and_metrics():
    rng = np.random.default_rng(2)
    t = rng.normal(size=(16, 16, 16))
    rec = 0.5 * t
    k, T = transfer(rec, t, box=40.0)
    assert np.allclose(T, 0.5)
    k2, Pab = cross_power(rec, t, box=40.0)
    assert np.array_equal(k, k2) and np.allclose(Pab, 0.5 * cross_power(t, t, box=40.0)[1])
    m = recover_metrics(rec, t, box=40.0)
    assert isinstance(m, RecoveryMetrics)
    assert np.allclose(m.r, 1.0) and np.allclose(m.power_ratio, 0.25)
    assert m.k_half == pytest.approx(m.k[-1])   # never drops below 0.5
    assert cell_corr(rec, t) == pytest.approx(1.0)
    assert cell_corr(t, -t) == pytest.approx(-1.0)


def test_k_half_edge_cases():
    k = np.array([0.1, 0.2, 0.3])
    assert k_half(k, np.array([0.4, 0.3, 0.2])) == 0.1
    assert k_half(k, np.array([0.9, 0.8, 0.7])) == 0.3
    assert k_half(k, np.array([0.9, 0.7, 0.3]), level=0.8) == pytest.approx(0.15)


def test_shape_validation():
    with pytest.raises(ValidationError):
        power_spectrum(np.zeros((4, 4, 5)), box=1.0)
    with pytest.raises(ValidationError):
        cross_r(np.zeros((4, 4, 4)), np.zeros((8, 8, 8)), box=1.0)
