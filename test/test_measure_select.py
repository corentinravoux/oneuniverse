"""measure T2 — select + clean."""
import sys
from pathlib import Path

from oneuniverse.measure.select import select_clean

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_select_clean_applies_zrange_and_quality(tmp_path):
    view = synthetic_point_view(tmp_path, n=4000, seed=2)
    cat = select_clean(view, z_range=(0.4, 0.7),
                        quality_column="quality", quality_min=1)
    assert cat["z"].between(0.4, 0.7).all()
    assert (cat["quality"] >= 1).all()
    assert cat["z"].notna().all()
    assert len(cat) < view.n_rows           # cuts removed rows
