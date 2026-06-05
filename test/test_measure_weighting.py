"""measure T3 — total weight assembly."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.weighting import assemble_weight

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_total_weight_is_product_of_components(tmp_path):
    view = synthetic_point_view(tmp_path, n=2000, seed=3)
    cat = select_clean(view, z_range=(0.1, 1.0))
    # FKPWeight.nbar is a CALLABLE z -> n̄(z); synthetic n̄ is constant 1e-3
    weights = [FKPWeight(nbar=lambda z: np.full_like(z, 1e-3), P0=1e4),
               ColumnWeight("weight_comp"), ColumnWeight("weight_sys")]
    out, recipe = assemble_weight(cat, weights)
    expected = (1.0 / (1.0 + 1e-3 * 1e4)
                * cat["weight_comp"].to_numpy() * cat["weight_sys"].to_numpy())
    assert np.allclose(out["weight"].to_numpy(), expected)
    assert "fkp" in recipe[0].lower()
    assert (out["weight"] > 0).all()
