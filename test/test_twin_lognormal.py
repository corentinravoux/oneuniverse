"""Lognormal tracer mock — preserves the (cross-)bias the Wiener filter uses."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.verify import _bin_kgrid, _bins


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


_BOX, _N = 256.0, 64


def _cross_bias(dg, truth):
    fk = np.fft.rfftn(dg); tk = np.fft.rfftn(truth)
    km = _bin_kgrid(_N, _BOX); idx, e, ctr = _bins(km, _BOX, _N)
    cross = np.real(fk * np.conj(tk)).ravel(); pt = (np.abs(tk) ** 2).ravel()
    rs = [cross[idx == i].sum() / pt[idx == i].sum()
          for i in range(1, 6) if (idx == i).sum()]
    return float(np.median(rs))


def test_lognormal_preserves_cross_bias_where_clip_fails():
    c = _cosmo()
    truth = generate_density_field(c, box_size=_BOX, n_grid=_N, z=0.0, seed=2)
    assert truth.std() > 1.5            # mildly non-linear regime
    clip = mock_tracer_field(truth, box_size=_BOX, nbar=5e-2, bias=1.5,
                             seed=3, model="clip")
    logn = mock_tracer_field(truth, box_size=_BOX, nbar=5e-2, bias=1.5,
                             seed=3, model="lognormal")
    b_clip = _cross_bias(clip["delta_g"], truth)
    b_logn = _cross_bias(logn["delta_g"], truth)
    assert b_clip < 1.0                 # clip biases low (~0.6)
    assert 1.25 < b_logn < 1.75         # lognormal preserves ~1.5


def test_lognormal_intensity_always_positive():
    c = _cosmo()
    truth = generate_density_field(c, box_size=_BOX, n_grid=_N, z=0.0, seed=2)
    out = mock_tracer_field(truth, box_size=_BOX, nbar=5e-2, bias=2.0, seed=1,
                            model="lognormal")
    assert out["counts"].min() >= 0     # Poisson of a positive rate, no clip


def test_clip_is_default_unchanged():
    c = _cosmo()
    truth = generate_density_field(c, box_size=_BOX, n_grid=_N, z=0.0, seed=2)
    a = mock_tracer_field(truth, box_size=_BOX, nbar=1e-2, bias=1.5, seed=4)
    b = mock_tracer_field(truth, box_size=_BOX, nbar=1e-2, bias=1.5, seed=4,
                          model="clip")
    np.testing.assert_array_equal(a["counts"], b["counts"])
