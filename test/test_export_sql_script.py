"""The export_to_sql CLI turns an OUF (or OUF-Sim) directory into a .sqlite."""
import sqlite3
import subprocess
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd


def _tiny_ouf(tmp_path: Path) -> Path:
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec
    n = 200
    rng = np.random.default_rng(0)
    ra = rng.uniform(150, 160, n); dec = rng.uniform(0, 10, n)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": rng.uniform(0.5, 2.0, n),
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n), "survey_id": np.zeros(n, "i8"),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    od = tmp_path / "toy" / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=od, survey_name="toy",
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name="toy", version="0"))
    return tmp_path / "toy"


def test_cli_exports_sqlite(tmp_path):
    src = _tiny_ouf(tmp_path)
    out = tmp_path / "toy.sqlite"
    r = subprocess.run(
        [sys.executable, "scripts/export_to_sql.py", str(src), "-o", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    con = sqlite3.connect(out)
    n = con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    assert n == 200
