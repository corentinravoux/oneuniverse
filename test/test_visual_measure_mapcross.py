"""measure map×catalog — the diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_mapcross_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_map_cross.png"
    assert p.is_file() and p.stat().st_size > 5_000
