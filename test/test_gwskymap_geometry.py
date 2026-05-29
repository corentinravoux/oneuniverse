"""Phase 22 T1/T4 — GW_SKYMAP geometry scaffold + round-trip."""
import healpy as hp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import (
    DEFAULT_PARTITION_ROWS,
    DataGeometry,
    GEOMETRY_COLUMNS,
    GW_SKYMAP_DATA_REQUIRED_COLUMNS,
    validate_columns,
)
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def test_gwskymap_enum_value():
    assert DataGeometry.GW_SKYMAP.value == "gw_skymap"


def test_gwskymap_in_geometry_columns():
    assert DataGeometry.GW_SKYMAP in GEOMETRY_COLUMNS


def test_gwskymap_required_columns_contents():
    cols = set(GW_SKYMAP_DATA_REQUIRED_COLUMNS)
    assert {"event_id", "event_name", "map_nside", "map_nest", "prob"} <= cols


def test_gwskymap_default_partition_rows_present():
    assert DataGeometry.GW_SKYMAP in DEFAULT_PARTITION_ROWS


def test_validate_columns_accepts_gwskymap_data():
    nside = 8
    df = pd.DataFrame({
        "event_id":   np.array([0], dtype="i8"),
        "event_name": np.array(["GW230529"], dtype=object),
        "map_nside":  np.array([nside], dtype="i4"),
        "map_nest":   np.array([True], dtype="bool"),
        "prob":       [np.zeros(hp.nside2npix(nside), dtype="f4")],
    })
    assert validate_columns(list(df.columns), DataGeometry.GW_SKYMAP, "data") == []


def test_gwskymap_writer_reader_roundtrip(tmp_path):
    nside = 8
    npix = hp.nside2npix(nside)
    df = pd.DataFrame({
        "event_id":   np.array([0, 1], dtype="i8"),
        "event_name": np.array(["GW230529", "GW230601"], dtype=object),
        "map_nside":  np.array([nside, nside], dtype="i4"),
        "map_nest":   np.array([True, True], dtype="bool"),
        "prob":       [
            np.zeros(npix, dtype="f4"),
            (np.linspace(0, 1, npix) / npix).astype("f4"),
        ],
    })
    out = tmp_path / "gw" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="gw", survey_type="gw",
        geometry=DataGeometry.GW_SKYMAP,
        loader=LoaderSpec(name="gw_fixture", version="0"),
        column_dtypes={"prob": "list<f4>"},
    )
    m = read_manifest(out / "manifest.json")
    assert m.geometry is DataGeometry.GW_SKYMAP
    parts = sorted(out.rglob("part_*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    out_probs = table.column("prob").to_pylist()
    assert [len(p) for p in out_probs] == [npix, npix]
