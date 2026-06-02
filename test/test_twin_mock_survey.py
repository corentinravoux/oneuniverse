"""Phase C4 — mock survey geometry / selection layer."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.mock_survey import (
    ball_mask,
    radial_completeness,
    slab_mask,
)
from oneuniverse.twin.wiener import wiener_reconstruct


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _smooth(f, box, R):
    n = f.shape[0]
    kx = np.fft.fftfreq(n, d=box / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box / n) * 2 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    return np.fft.irfftn(np.fft.rfftn(f) * np.exp(-0.5 * k2 * R ** 2),
                         s=(n, n, n))


def test_slab_mask_fraction_and_axis():
    m = slab_mask(64, frac=0.25, axis=2)
    assert m.shape == (64, 64, 64)
    assert abs(m.mean() - 0.25) < 0.02
    assert m[:, :, 0].all() and not m[:, :, -1].any()


def test_ball_mask_inside_only():
    m = ball_mask(64, box_size=256.0, radius=80.0)
    assert set(np.unique(m)) <= {0.0, 1.0}
    # roughly the sphere volume fraction
    frac = (4 / 3) * np.pi * 80.0 ** 3 / 256.0 ** 3
    assert abs(m.mean() - frac) < 0.05


def test_radial_completeness_declines():
    w = radial_completeness(64, box_size=256.0, r_scale=80.0)
    assert w.min() >= 0.0 and w.max() <= 1.0 + 1e-9
    centre = w[32, 32, 32]
    corner = w[0, 0, 0]
    assert centre > corner          # completeness highest near the observer


def test_masked_reconstruction_recovers_inside_footprint():
    c = _cosmo()
    box, n = 256.0, 64
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=2)
    mask = ball_mask(n, box_size=box, radius=90.0)
    obs = mock_tracer_field(truth, box_size=box, nbar=5e-2, bias=1.5,
                            seed=3, mask=mask)
    rec = wiener_reconstruct(obs["delta_g"], c, box_size=box, nbar=5e-2,
                             bias=1.5)
    inside = mask > 0
    ts = _smooth(truth, box, 16.0)[inside]
    rs = _smooth(rec, box, 16.0)[inside]
    # large-scale structure recovered inside the survey footprint
    assert np.corrcoef(ts, rs)[0, 1] > 0.6
