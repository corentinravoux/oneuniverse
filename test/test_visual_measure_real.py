"""measure — the real eBOSS DR16Q diagnostic figure (committed artifact)."""
from pathlib import Path


def test_real_eboss_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_real_eboss.png"
    assert p.is_file() and p.stat().st_size > 5_000
