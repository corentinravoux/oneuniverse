"""measure PV/SN-T3 — light-curve carrier from LIGHTCURVE geometry."""
import sys
from pathlib import Path

from oneuniverse.measure.lightcurve import LightcurveSet, lightcurves_from_view

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_lightcurve_view  # noqa: E402


def test_lightcurves_from_view(tmp_path):
    view = synthetic_lightcurve_view(tmp_path, n_obj=20, n_epoch=8, seed=2)
    lc = lightcurves_from_view(view)
    assert isinstance(lc, LightcurveSet)
    assert lc.n_objects == 20
    one = lc.for_object(lc.object_ids[0])
    assert {"mjd", "flux", "filter"} <= set(one.columns)
    assert len(one) == 8
