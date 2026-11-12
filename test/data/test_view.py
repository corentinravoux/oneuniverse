import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.synthetic import write_synthetic_dataset

from oneuniverse._core.errors import FormatError, SchemaError, ValidationError
from oneuniverse._core.geometry import Cone, SkyPatch
from oneuniverse.data import DatasetView
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.schema import Geometry
from oneuniverse.data.view import cone_expr, patch_expr
from oneuniverse.data.writer import write_dataset


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "ds"
    write_synthetic_dataset(p, 100_000, 0, partition_nside=8)
    return DatasetView.from_path(p)


def test_properties(ds):
    assert ds.n_rows == 100_000 and ds.nside == 8 and ds.n_partitions == ds.index.num_rows
    assert "mag" in ds.columns and "cell" not in ds.columns


def test_cone_scan_matches_bruteforce(ds):
    cone = Cone(185.0, 15.0, 10.0)
    full = ds.table(columns=["ra", "dec", "object_id"])
    exp = full.filter(pa.array(cone.contains(full["ra"].to_numpy(), full["dec"].to_numpy())))
    got = ds.table(cone=cone, columns=["ra", "dec", "object_id"])
    assert sorted(got["object_id"].to_pylist()) == sorted(exp["object_id"].to_pylist())
    assert len(ds.cells_for(cone=cone)) < ds.n_partitions / 4


def test_view_cone_at_ra_zero(ds):
    cone = Cone(0.2, 0.0, 2.0)
    full = ds.table(columns=["ra", "dec"])
    n = int(cone.contains(full["ra"].to_numpy(), full["dec"].to_numpy()).sum())
    assert ds.table(cone=cone).num_rows == n and n > 0
    assert len(ds.cells_for(cone=cone)) < len(np.unique(ds.index["cell"].to_numpy()))


def test_view_patch_wrapping(ds):
    p = SkyPatch(350.0, 10.0, -5.0, 5.0)
    full = ds.table(columns=["ra", "dec"])
    n = int(p.contains(full["ra"].to_numpy(), full["dec"].to_numpy()).sum())
    assert ds.table(patch=p).num_rows == n and n > 0
    assert len(ds.cells_for(patch=p)) < len(np.unique(ds.index["cell"].to_numpy()))


def test_exprs_match_numpy_at_boundary():
    cone = Cone(10.0, 20.0, 5.0)
    ra = np.array([10.0, 15.0, 10.0, 12.0, 300.0]); dec = np.array([20.0, 20.0, 25.0, 22.0, -40.0])
    t = pa.table({"ra": ra, "dec": dec})
    got = t.filter(cone_expr(cone))
    assert got.num_rows == int(cone.contains(ra, dec).sum())
    p = SkyPatch(350.0, 360.0, -5.0, 5.0)
    ra2 = np.array([-10.0, 0.0, 360.0, 355.0, 20.0]); dec2 = np.zeros(5)
    assert pa.table({"ra": ra2, "dec": dec2}).filter(patch_expr(p)).num_rows == int(p.contains(ra2, dec2).sum())


def test_z_and_ranges_prune_and_filter(ds):
    t = ds.table(z=(0.5, 0.6), ranges={"mag": (18.0, 19.0)}, columns=["z", "mag"])
    assert t.num_rows > 0 and pc.min(t["z"]).as_py() >= 0.5 and pc.max(t["mag"]).as_py() <= 19.0
    assert pc.min(t["mag"]).as_py() >= 18.0 and pc.max(t["z"]).as_py() <= 0.6


def test_ranges_unknown_column_and_t_without_extension(ds):
    with pytest.raises(ValidationError):
        ds.table(ranges={"nope": (0, 1)})
    with pytest.raises(FormatError, match="temporal"):
        ds.table(t=(0.0, 1.0))


def test_user_filter_is_anded(ds):
    t = ds.table(z=(0.5, 0.6), filter=pc.field("mag") < 17.0, columns=["z", "mag"])
    assert t.num_rows > 0 and pc.max(t["mag"]).as_py() < 17.0 and pc.min(t["z"]).as_py() >= 0.5


def test_empty_result_keeps_schema(ds):
    t = ds.table(z=(5.0, 6.0))
    assert t.num_rows == 0 and t.schema.equals(ds.table(columns=None).schema)


def test_take_returns_rows_in_order(ds):
    pid, tab = next(ds.iter_partitions(columns=["object_id"]))
    ids = tab["object_id"].to_numpy()
    got = ds.take(np.array([pid, pid], np.int32), np.array([5, 2]), columns=["object_id"])
    assert got["object_id"].to_pylist() == [ids[5], ids[2]]


def test_take_interleaved_and_errors(ds):
    ids = {p: t["object_id"].to_numpy() for p, t in ds.iter_partitions(columns=["object_id"])
           if p < 2}
    part = np.array([1, 0, 1, 0, 1], np.int32); off = np.array([3, 1, 3, 0, 2])
    got = ds.take(part, off, columns=["object_id"])["object_id"].to_pylist()
    assert got == [ids[p][o] for p, o in zip(part, off)]
    empty = ds.take(np.array([], np.int32), np.array([], np.int64), columns=["ra", "z"])
    assert empty.num_rows == 0 and empty.schema.names == ["ra", "z"]
    with pytest.raises(ValidationError):
        ds.take(np.array([ds.n_partitions]), np.array([0]))
    with pytest.raises(ValidationError):
        ds.take(np.array([0]), np.array([int(ds.index["n_rows"][0].as_py())]))


def test_read_returns_dataframe_with_categoricals(ds):
    df = ds.read(z=(0.1, 0.2), columns=["z", "survey", "z_type"])
    assert isinstance(df, pd.DataFrame) and str(df["survey"].dtype) == "category"


def test_scan_is_lazy_batches(ds):
    b = next(ds.scan(columns=["ra"], batch_size=1000))
    assert isinstance(b, pa.RecordBatch) and b.num_rows <= 1000


def test_load_pdf_requires_extension(ds):
    with pytest.raises(FormatError, match="pdf"):
        ds.load_pdf()


def test_sightline_geometry_round_trip(tmp_path):
    n = 300; rng = np.random.default_rng(0); lam = [np.linspace(3600, 5000, 50).astype(np.float32)] * n
    t = pa.table({"sightline_id": np.arange(n), "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-30, 30, n), "z": rng.uniform(2, 3, n), "survey": ["lya"] * n, "source_row": np.arange(n),
                  "wavelength": pa.array(lam, pa.list_(pa.float32())), "delta": pa.array([rng.normal(0, 0.1, 50).astype(np.float32)] * n, pa.list_(pa.float32())), "weight": pa.array([np.ones(50, np.float32)] * n, pa.list_(pa.float32()))})
    m = write_dataset(t, tmp_path / "sl", geometry=Geometry.SIGHTLINE, survey=SurveyMeta("lya", "lyman_alpha"), loader=LoaderInfo("t", "1", {}))
    v = DatasetView.from_path(tmp_path / "sl"); assert m.geometry == Geometry.SIGHTLINE and v.n_rows == n and v.table(columns=["delta"])["delta"].type == pa.list_(pa.float32())
    with pytest.raises(SchemaError, match="wavelength"): write_dataset(t.drop(["wavelength"]), tmp_path / "bad", geometry=Geometry.SIGHTLINE, survey=SurveyMeta("lya", "lyman_alpha"), loader=LoaderInfo("t", "1", {}))


# ---- Task 5 review follow-ups ---------------------------------------
def _boundary_points(rng, cone, n):
    """Points at angular distance exactly ``cone.radius`` (up to rounding) from the centre."""
    ra0, dec0, r = np.radians(cone.ra), np.radians(cone.dec), np.radians(cone.radius)
    b = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(np.sin(dec0) * np.cos(r) + np.cos(dec0) * np.sin(r) * np.cos(b))
    ra = ra0 + np.arctan2(np.sin(b) * np.sin(r) * np.cos(dec0),
                          np.cos(r) - np.sin(dec0) * np.sin(dec))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def test_cone_expr_boundary_agrees_with_cone_contains():
    rng = np.random.default_rng(7)
    tot = bad = 0
    for _ in range(100):
        cone = Cone(rng.uniform(0, 360), rng.uniform(-80, 80), rng.uniform(0.5, 30))
        ra, dec = _boundary_points(rng, cone, 2_000)
        tab = pa.table({"ra": ra, "dec": dec, "i": np.arange(ra.size)})
        got = np.zeros(ra.size, bool)
        got[tab.filter(cone_expr(cone))["i"].to_numpy()] = True
        bad += int(np.sum(got != cone.contains(ra, dec))); tot += ra.size
    assert bad / tot <= 1e-4, (bad, tot)  # >= 99.99 % (ulp-level sin differences)


def _multi_rg_dataset(tmp_path, monkeypatch, with_time=False, n=6_000):
    import numpy as _np
    from oneuniverse.data import writer
    monkeypatch.setattr(writer, "_ROW_GROUP", 32)
    monkeypatch.setattr(writer, "MAX_ROWS_PER_FILE", 32 * 1000)
    from fixtures.synthetic import synthetic_point_table
    t = synthetic_point_table(n, 1)
    ext = None
    if with_time:
        t = t.append_column("mjd", pa.array(_np.linspace(50_000, 60_000, n)))
        ext = {"temporal": {"version": 1, "time_column": "mjd"}}
    write_dataset(t, tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"),
                  loader=LoaderInfo("t", "1", {}), partition_nside=1,
                  extra_stats_columns=["mjd"] if with_time else [], extensions=ext)
    return DatasetView.from_path(tmp_path / "ds")


def test_take_reads_only_needed_row_groups(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    v = _multi_rg_dataset(tmp_path, monkeypatch)
    sizes = v.index["n_rows"].to_numpy(); p = int(np.argmax(sizes))
    pf = pq.ParquetFile(v.path / v.index["path"][p].as_py())
    assert pf.metadata.num_row_groups > 2
    ids = pf.read(columns=["object_id"])["object_id"].to_numpy()
    off = np.array([100, 5, 100, 200])  # three groups, with a repeat
    calls = []
    orig = pq.ParquetFile.read_row_groups
    monkeypatch.setattr(pq.ParquetFile, "read_row_groups",
                        lambda self, rgs, **kw: calls.append(list(rgs)) or orig(self, rgs, **kw))
    monkeypatch.setattr(pq.ParquetFile, "read",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("whole-file read")))
    got = v.take(np.full(off.size, p, np.int32), off, columns=["object_id"])
    assert got["object_id"].to_numpy().tolist() == ids[off].tolist()
    assert calls and all(len(c) < pf.metadata.num_row_groups for c in calls)
    one = v.take(np.array([p]), np.array([40]), columns=["object_id"])
    assert one["object_id"].to_pylist() == [ids[40]] and calls[-1] == [1]


def test_take_rejects_non_integer_indices(ds):
    with pytest.raises(ValidationError):
        ds.take(np.array([0.0]), np.array([1]))
    with pytest.raises(ValidationError):
        ds.take(np.array([0]), np.array([1.5]))
    with pytest.raises(ValidationError):
        ds.take(np.array([True]), np.array([1]))


def test_t_selection_with_temporal_extension(tmp_path, monkeypatch):
    v = _multi_rg_dataset(tmp_path, monkeypatch, with_time=True)
    full = v.table(columns=["mjd"])["mjd"].to_numpy()
    got = v.table(t=(52_000.0, 53_000.0), columns=["mjd"])["mjd"].to_numpy()
    assert got.size == np.sum((full >= 52_000) & (full <= 53_000)) > 0
    assert got.min() >= 52_000 and got.max() <= 53_000
