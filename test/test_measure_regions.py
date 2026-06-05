"""measure T7 — HEALPix region assignment."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_region_ids_are_stable_and_shared(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=7)
    cat = select_clean(view, z_range=(0.1, 1.0))
    r1 = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(), nside=4)
    r2 = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(), nside=4)
    assert r1.dtype == np.int64 and len(r1) == len(cat)
    np.testing.assert_array_equal(r1, r2)          # deterministic
    assert r1.min() >= 0
    assert len(np.unique(r1)) > 1                   # patch spans >1 region
