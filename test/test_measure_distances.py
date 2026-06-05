"""measure PV/SN-T1 — distance-indicator atoms."""
import sys
from pathlib import Path

import pytest

from oneuniverse.measure.distances import attach_distances

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_pv_view  # noqa: E402


def test_attach_distances_validates(tmp_path):
    view = synthetic_pv_view(tmp_path, n=1500, seed=1)
    cat = view.read()
    out, prov = attach_distances(cat, columns=("mu", "mu_err", "v_pec",
                                               "sigma_v"))
    assert {"mu", "v_pec", "sigma_v"} <= set(out.columns)
    assert "mu" in prov
    with pytest.raises(ValueError, match="distance column"):
        attach_distances(cat.drop(columns=["v_pec"]), columns=("mu", "v_pec"))
