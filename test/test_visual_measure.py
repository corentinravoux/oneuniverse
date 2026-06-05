"""measure T9 — the galaxy-clustering diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_measure_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_galaxy_clustering.png"
    assert p.is_file() and p.stat().st_size > 5_000
