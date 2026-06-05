"""measure Lyα — the diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_lya_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_lya.png"
    assert p.is_file() and p.stat().st_size > 5_000
