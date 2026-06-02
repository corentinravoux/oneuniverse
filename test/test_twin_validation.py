"""Phase C3 — standard recovery-metrics harness."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.validation import RecoveryMetrics, recover_metrics


def _truth():
    c = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
    return generate_density_field(c, box_size=256.0, n_grid=64, z=0.0, seed=2)


def test_identical_fields_perfect_recovery():
    t = _truth()
    m = recover_metrics(t, t, box_size=256.0)
    assert isinstance(m, RecoveryMetrics)
    good = np.isfinite(m.r)
    assert np.all(m.r[good] > 0.999)
    assert np.allclose(m.transfer[np.isfinite(m.transfer)], 1.0, atol=1e-6)
    assert np.isnan(m.k_half)          # r never drops below 0.5


def test_scaled_field_transfer_is_scale():
    t = _truth()
    m = recover_metrics(0.5 * t, t, box_size=256.0)
    good = np.isfinite(m.transfer)
    assert np.allclose(m.transfer[good], 0.5, atol=1e-6)
    assert np.all(m.r[np.isfinite(m.r)] > 0.999)   # still perfectly correlated


def test_more_noise_lowers_k_half():
    # r(k) is insensitive to a deterministic filter (smoothing keeps r=1);
    # it is *noise* that decorrelates. More noise -> r drops sooner -> the
    # reconstruction scale k_half moves to lower k.
    t = _truth()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(t.shape)
    light = recover_metrics(t + 5.0 * noise, t, box_size=256.0)
    heavy = recover_metrics(t + 20.0 * noise, t, box_size=256.0)
    assert np.isfinite(light.k_half) and np.isfinite(heavy.k_half)
    assert heavy.k_half < light.k_half
    # robust secondary check: more noise -> lower correlation at high k
    band = light.k > 0.3
    assert np.nanmedian(heavy.r[band]) < np.nanmedian(light.r[band])
