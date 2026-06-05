"""measure WL-T5 — 3x2pt bundle."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.lensing import build_3x2pt

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view, synthetic_shear_view  # noqa: E402


def test_build_3x2pt_shares_region_and_pairs(tmp_path):
    lens = synthetic_point_view(tmp_path, n=4000, seed=4, name="lens")
    src = synthetic_shear_view(tmp_path, n=4000, seed=5, kind="metacal",
                               with_pdf=True, n_tomo=2, name="src")
    ms = build_3x2pt(lens, src, z_grid=np.linspace(0, 2, 41), nside_region=4,
                     lens_z_range=(0.2, 0.6),
                     lens_weights_columns=("weight_comp",))
    assert set(ms.products) == {"lens", "src"}
    assert ms.metadata.nside_region == 4
    assert ("lens", "src") in ms.spec.pairs
    assert ms.spec.pair_statistics[("lens", "src")] == "gamma_t"
    assert ms.spec.pair_statistics[("src", "src")] == "xi_pm"
    ms.check_invariants()
