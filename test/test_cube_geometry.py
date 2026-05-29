"""Phase 22 T1/T4 — CUBE geometry scaffold + round-trip."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import (
    CUBE_DATA_REQUIRED_COLUMNS,
    DEFAULT_PARTITION_ROWS,
    DataGeometry,
    GEOMETRY_COLUMNS,
    validate_columns,
)
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def test_cube_enum_value():
    assert DataGeometry.CUBE.value == "cube"


def test_cube_in_geometry_columns():
    assert DataGeometry.CUBE in GEOMETRY_COLUMNS
    assert "data" in GEOMETRY_COLUMNS[DataGeometry.CUBE]


def test_cube_required_columns_contents():
    cols = set(CUBE_DATA_REQUIRED_COLUMNS)
    assert {"cube_id", "ra", "dec", "shape", "cube"} <= cols


def test_cube_default_partition_rows_present():
    assert DataGeometry.CUBE in DEFAULT_PARTITION_ROWS


def test_validate_columns_accepts_cube_data():
    df = pd.DataFrame({
        "cube_id": np.array([0], dtype="i8"),
        "ra":      np.array([10.0], dtype="f8"),
        "dec":     np.array([0.0], dtype="f8"),
        "shape":   [np.array([3, 3, 4], dtype="i4")],
        "cube":    [np.zeros(36, dtype="f4")],
    })
    assert validate_columns(list(df.columns), DataGeometry.CUBE, "data") == []


def test_cube_writer_reader_roundtrip(tmp_path):
    n_cubes = 3
    shape = (3, 3, 4)
    npx = shape[0] * shape[1] * shape[2]
    df = pd.DataFrame({
        "cube_id": np.arange(n_cubes, dtype="i8"),
        "ra":  np.linspace(10.0, 12.0, n_cubes).astype("f8"),
        "dec": np.zeros(n_cubes, dtype="f8"),
        "shape": [np.array(shape, dtype="i4") for _ in range(n_cubes)],
        "cube":  [
            np.arange(npx, dtype="f4") + i * npx
            for i in range(n_cubes)
        ],
    })
    out = tmp_path / "ifu" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="ifu", survey_type="ifu",
        geometry=DataGeometry.CUBE,
        loader=LoaderSpec(name="ifu_fixture", version="0"),
        column_dtypes={"cube": "list<f4>", "shape": "i4[3]"},
    )
    m = read_manifest(out / "manifest.json")
    assert m.geometry is DataGeometry.CUBE
    parts = sorted((out).rglob("part_*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    out_cubes = table.column("cube").to_pylist()
    assert [len(c) for c in out_cubes] == [npx, npx, npx]
