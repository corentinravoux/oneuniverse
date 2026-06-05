"""measure WL-T6 — the cosmic-shear diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_wl_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_weak_lensing.png"
    assert p.is_file() and p.stat().st_size > 5_000
