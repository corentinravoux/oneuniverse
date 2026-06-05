"""measure T6 — randoms ingest + generate."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.nz import nz_from_spec_z
from oneuniverse.measure.randoms import generate_randoms, randoms_from_view
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.window import footprint_from_positions

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_generate_randoms_inside_window_and_nz(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=6)
    cat = select_clean(view, z_range=(0.1, 1.0))
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=64)
    nz = nz_from_spec_z(cat["z"].to_numpy(), edges=np.linspace(0.0, 1.2, 25))
    rnd, source = generate_randoms(win, nz, n_randoms=20000, seed=1)
    assert source == "generated"
    assert win.contains(rnd["ra"].to_numpy(), rnd["dec"].to_numpy()).all()
    assert rnd["z"].min() >= 0.0 and rnd["z"].max() <= 1.2
    assert len(rnd) == 20000 and (rnd["weight"] == 1.0).all()


def test_ingest_randoms_from_view(tmp_path):
    rview = synthetic_point_view(tmp_path, n=5000, seed=99, name="rand")
    rnd, source = randoms_from_view(rview)
    assert source == "ingested" and len(rnd) == 5000
    assert {"ra", "dec", "z"} <= set(rnd.columns)
