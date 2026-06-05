"""measure PV/SN-T2 — lazy row-correlated covariance handle."""
import numpy as np

from oneuniverse.measure.covariance import CovarianceHandle


def test_covariance_handle_lazy_load(tmp_path):
    cov = np.diag(np.arange(1, 6, dtype=float))
    p = tmp_path / "cov.npy"; np.save(p, cov)
    h = CovarianceHandle(cov_id="sn5", path=str(p), n=5)
    assert h.n == 5
    assert not h.is_loaded
    mat = h.matrix()
    assert mat.shape == (5, 5) and h.is_loaded
    np.testing.assert_allclose(np.diag(mat), np.arange(1, 6))
