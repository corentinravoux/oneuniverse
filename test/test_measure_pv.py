"""measure PV/SN-T4 — peculiar-velocity MeasurementSet."""
import sys
from pathlib import Path

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.pvsn import build_peculiar_velocity

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_pv_view  # noqa: E402


def test_build_peculiar_velocity(tmp_path):
    view = synthetic_pv_view(tmp_path, n=3000, seed=3)
    ms = build_peculiar_velocity(
        view, tracer="pv", z_range=(0.0, 0.1),
        distance_columns=("mu", "mu_err", "v_pec", "sigma_v"), nside_region=4)
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["pv"]
    assert {"v_pec", "sigma_v"} <= set(ps.catalog.columns)
    assert ms.spec.estimator_family == "velocity"
    assert "v_pec" in ps.provenance.extra["distance_columns"]
    ms.check_invariants()
