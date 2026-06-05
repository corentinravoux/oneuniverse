"""measure T4 — HEALPix footprint window."""
import sys
from pathlib import Path

from oneuniverse.measure.select import select_clean
from oneuniverse.measure.window import Window, footprint_from_positions

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_footprint_covers_data_pixels(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=4)
    cat = select_clean(view, z_range=(0.1, 1.0))
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=64)
    assert isinstance(win, Window)
    assert win.mask.sum() > 0
    assert win.contains(cat["ra"].to_numpy(), cat["dec"].to_numpy()).all()
    assert 0.0 < win.covered_fraction() < 1.0
