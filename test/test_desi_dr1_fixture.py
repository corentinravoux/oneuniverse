"""Smoke test for the DR1 QSO synthetic FITS fixture."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits  # noqa: E402


def test_fake_fits_has_all_loader_columns(tmp_path):
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
        _FITS_COLUMNS,
    )
    out = write_fake_desi_dr1_fits(tmp_path, n_rows=200, seed=0)
    import fitsio
    with fitsio.FITS(str(out)) as f:
        cols = set(f[1].get_colnames())
    for c in _FITS_COLUMNS:
        assert c in cols, f"fixture missing {c!r}"
