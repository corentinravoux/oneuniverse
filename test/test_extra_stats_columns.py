"""Phase 17 T5 — writer populates extra_ranges per partition."""
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def _base_core(n: int) -> pd.DataFrame:
    ra = np.linspace(0.0, 30.0, n).astype("f8")
    dec = np.linspace(-5.0, 5.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.linspace(0.1, 0.9, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_extra_ranges_present_per_partition(tmp_path):
    n = 200
    df = _base_core(n)
    df["snr"] = np.linspace(5.0, 95.0, n).astype("f4")
    df["ebv"] = np.linspace(0.0, 0.1, n).astype("f4")
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        extra_stats_columns=["snr", "ebv"],
    )
    m = read_manifest(out / "manifest.json")
    have_snr = False
    have_ebv = False
    for p in m.partitions:
        er = p.stats.extra_ranges
        if "snr" in er:
            lo, hi = er["snr"]
            assert lo <= hi
            have_snr = True
        if "ebv" in er:
            have_ebv = True
    assert have_snr and have_ebv
