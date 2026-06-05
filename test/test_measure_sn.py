"""measure PV/SN-T5 — SN Hubble MeasurementSet with covariance handle."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.covariance import CovarianceHandle
from oneuniverse.measure.pvsn import build_sn_hubble

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_sn_view  # noqa: E402


def test_build_sn_hubble_with_cov(tmp_path):
    view, n = synthetic_sn_view(tmp_path, n=200, seed=4)
    cov = np.diag(np.full(n, 0.01))
    p = tmp_path / "sncov.npy"; np.save(p, cov)
    ms = build_sn_hubble(view, tracer="sn", z_range=(0.0, 1.5),
                         distance_columns=("mu", "mu_err"),
                         covariance=CovarianceHandle("sn", str(p), n),
                         nside_region=2)
    ps = ms.products["sn"]
    assert {"mu", "z"} <= set(ps.catalog.columns)
    assert ms.spec.statistic == "hubble"
    assert ps.provenance.extra["cov_id"] == "sn"
    ms.check_invariants()
