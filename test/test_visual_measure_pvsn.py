"""measure PV/SN-T6 — the PV/SN diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_pvsn_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_pv_sn.png"
    assert p.is_file() and p.stat().st_size > 5_000
