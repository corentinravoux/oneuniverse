"""S17 T8 — the scaling diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_s17_figure_exists():
    p = Path(__file__).parent / "test_output" / "s17_scaling.png"
    assert p.is_file() and p.stat().st_size > 5_000
