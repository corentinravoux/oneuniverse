"""measure WL-T4 — cosmic shear MeasurementSet."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.lensing import build_cosmic_shear

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402


def test_build_cosmic_shear(tmp_path):
    view = synthetic_shear_view(tmp_path, n=4000, seed=3, kind="metacal",
                                with_pdf=True, n_tomo=2)
    ms = build_cosmic_shear(view, tracer="src", kind="metacal",
                            tomo_column="tomo_bin",
                            z_grid=np.linspace(0, 2, 41), nside_region=4)
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["src"]
    assert {"e1", "e2", "weight"} <= set(ps.catalog.columns)
    assert ps.photoz is not None
    assert isinstance(ps.nz, dict) and set(ps.nz) == {0, 1}
    assert ms.spec.statistic == "xi_pm"
    assert ms.spec.estimator_family == "lensing"
    ms.check_invariants()
