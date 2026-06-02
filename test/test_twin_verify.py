"""Phase C1 T3 — cross-correlation r(k)."""
import numpy as np

from oneuniverse.twin.verify import cross_correlation, power_ratio


def test_self_correlation_is_one():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((32, 32, 32))
    k, r = cross_correlation(f, f, box_size=200.0)
    assert np.all(r[np.isfinite(r)] > 0.999)


def test_uncorrelated_fields_near_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32, 32))
    b = rng.standard_normal((32, 32, 32))
    k, r = cross_correlation(a, b, box_size=200.0)
    assert np.nanmedian(np.abs(r)) < 0.2


def test_power_ratio_self_is_one():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((32, 32, 32))
    k, ratio = power_ratio(f, f, box_size=200.0)
    assert np.allclose(ratio[np.isfinite(ratio)], 1.0, atol=1e-6)
