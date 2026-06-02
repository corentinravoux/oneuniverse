"""Phase C1 T4 — end-to-end mock challenge."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.twin.mock_challenge import run_mock_challenge


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_recovers_large_scales():
    res = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                             nbar=5e-2, bias=1.5, seed=11)
    k, r = res["k"], res["r"]
    low = r[k < 0.05]
    high = r[k > 0.3]
    assert np.nanmedian(low) > 0.8
    assert np.nanmedian(high) < np.nanmedian(low)


def test_denser_survey_recovers_more():
    lo = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                            nbar=5e-3, bias=1.5, seed=11)
    hi = run_mock_challenge(_cosmo(), box_size=256.0, n_grid=64,
                            nbar=1e-1, bias=1.5, seed=11)
    band = (lo["k"] > 0.1) & (lo["k"] < 0.3)
    assert np.nanmedian(hi["r"][band]) > np.nanmedian(lo["r"][band])
