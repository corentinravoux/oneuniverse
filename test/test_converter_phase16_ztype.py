"""Phase 16 T7 — writer validates z_type and records observed values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def _df(z_types):
    n = len(z_types)
    import healpy as hp
    ra = np.linspace(0.0, 1.0, n).astype("f8")
    dec = np.linspace(0.0, 1.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": np.full(n, 0.5, dtype="f4"),
        "z_type": np.array(z_types, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fixture"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_writer_records_observed_z_types(tmp_path):
    df = _df(["spec", "spec", "phot", "none"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="fixture", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fixture_loader", version="0"),
    )
    m = read_manifest(out / "manifest.json")
    assert set(m.observed_z_types) == {"spec", "phot", "none"}


def test_writer_rejects_unknown_z_type(tmp_path):
    df = _df(["spec", "made_up"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="unregistered"):
        write_ouf_dataset(
            df=df, out_dir=out,
            survey_name="fixture", survey_type="spectroscopic",
            geometry=DataGeometry.POINT,
            loader=LoaderSpec(name="fixture_loader", version="0"),
        )


def test_writer_accepts_newly_registered_z_type(tmp_path):
    from oneuniverse.data.ztypes import register_z_type

    register_z_type("spec_lya", description="Lyman-alpha z")
    df = _df(["spec_lya", "spec_lya"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="fixture", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fixture_loader", version="0"),
    )
    m = read_manifest(out / "manifest.json")
    assert "spec_lya" in m.observed_z_types
