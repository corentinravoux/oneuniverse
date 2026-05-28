"""Phase 17 T3 — write_ouf_dataset round-trips variable-length payloads."""
import healpy as hp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def _base_core(n: int) -> pd.DataFrame:
    ra = np.linspace(0.0, 10.0, n).astype("f8")
    dec = np.linspace(-5.0, 5.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": np.full(n, 0.5, dtype="f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_writer_emits_variable_length_list(tmp_path):
    df = _base_core(4)
    df["delta"] = [np.arange(k + 3, dtype="f4") for k in range(4)]
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        column_dtypes={"delta": "list<f4>"},
    )
    view = DatasetView.from_path(out.parent)
    out_df = view.read()
    lengths = [len(x) for x in out_df["delta"]]
    assert sorted(lengths) == [3, 4, 5, 6]


def test_writer_emits_fixedsize_bitweights(tmp_path):
    df = _base_core(3)
    df["BITWEIGHTS"] = [np.arange(64, dtype="i8")] * 3
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        column_dtypes={"BITWEIGHTS": "i8[64]"},
    )
    paths = sorted((out / "data").rglob("*.parquet"))
    assert paths, "no partition written"
    table = pq.read_table(paths[0])
    t = table.schema.field("BITWEIGHTS").type
    assert t.list_size == 64
