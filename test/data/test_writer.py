import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from fixtures.synthetic import (
    synthetic_point_table,
    synthetic_sightline_table,
    write_synthetic_dataset,
    write_synthetic_sightlines,
)

from oneuniverse._core.errors import FormatError, SchemaError, ValidationError
from oneuniverse._core.healpix import coarsen
from oneuniverse.data import writer
from oneuniverse.data.manifest import (
    LoaderInfo,
    OriginalFile,
    SurveyMeta,
    read_manifest,
    read_partition_index,
)
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import auto_partition_nside, write_dataset

S, L = SurveyMeta("s", "test"), LoaderInfo("t", "1", {})


def _w(data, out, **kw):
    return write_dataset(data, out, geometry=Geometry.POINT, survey=S, loader=L, **kw)


def test_layout_and_manifest(tmp_path):
    m = write_synthetic_dataset(tmp_path / "ds", 20_000, 0)
    assert (tmp_path / "ds/manifest.json").exists() and (tmp_path / "ds/_index.parquet").exists()
    files = sorted((tmp_path / "ds/data").rglob("*.parquet"))
    assert files and all(f.parent.name.startswith("cell=") for f in files)
    assert not (tmp_path / "ds/stage").exists()
    assert m.n_rows == 20_000 and m.partitioning.nside == auto_partition_nside(20_000)
    assert m.format_version == "3.0.0" and m.validity.version == 1
    assert m.created_utc.endswith("Z")
    assert read_manifest(tmp_path / "ds/manifest.json") == m
    assert pq.read_schema(files[0]).remove_metadata().equals(m.schema.remove_metadata())


def test_auto_partition_nside_values():
    assert auto_partition_nside(10) == 1
    assert auto_partition_nside(5_000 * 12 * 64) == 8
    assert auto_partition_nside(10**9) == 32


def test_index_stats_match_data(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 20_000, 1, partition_nside=4)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    assert idx.num_rows > 1
    for row in idx.to_pylist():
        t = pq.read_table(tmp_path / "ds" / row["path"], columns=["ra", "dec", "z", "hpix32"])
        assert t.num_rows == row["n_rows"]
        assert pc.min(t["z"]).as_py() == pytest.approx(row["z_min"])
        assert pc.max(t["ra"]).as_py() == pytest.approx(row["ra_max"])
        assert pc.min(t["dec"]).as_py() == pytest.approx(row["dec_min"])
        assert set(coarsen(t["hpix32"].to_numpy(), 32, 4)) == {row["cell"]}
        assert row["path"] == f"data/cell={row['cell']}/{row['path'].rsplit('/', 1)[1]}"
    assert sum(idx["n_rows"].to_pylist()) == 20_000


def test_cell_projection_equals_coarsen():
    hp = np.random.default_rng(0).integers(0, 12 * 32**2, 10_000).astype(np.int32)
    for nside in (1, 2, 4, 8, 16, 32):
        shift = 2 * int(np.log2(32 // nside))
        got = pc.cast(pc.shift_right(pa.array(hp), shift), pa.int32()).to_numpy()
        np.testing.assert_array_equal(got, coarsen(hp, 32, nside))


def test_extra_stats_columns(tmp_path):
    tab = synthetic_point_table(5_000, 2)
    _w(tab, tmp_path / "ds", extra_stats_columns=["mag"])
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    assert "mag_min" in idx.column_names and "mag_max" in idx.column_names
    assert min(idx["mag_min"].to_pylist()) == pytest.approx(float(pc.min(tab["mag"]).as_py()))
    with pytest.raises(ValidationError, match="nope"):
        _w(tab, tmp_path / "x", extra_stats_columns=["nope"])


_MEM_SCRIPT = """
import resource, sys, pathlib
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from fixtures.synthetic import synthetic_point_table
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta, read_manifest
from oneuniverse.data import writer
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import write_dataset
synthetic_point_table(10, 0)
def gen():
    for i in range(int(sys.argv[4])):
        yield synthetic_point_table(50_000, i).to_batches()[0]
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
write_dataset(gen(), pathlib.Path(sys.argv[3]), geometry=Geometry.POINT,
              survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {}))
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024,
      read_manifest(pathlib.Path(sys.argv[3]) / "manifest.json").n_rows)
"""


def _rss_growth_mib(tmp_path, n_batches):
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, "-c", _MEM_SCRIPT, str(root), str(root / "test"),
         str(tmp_path / "ds"), str(n_batches)],
        capture_output=True, text=True, check=True)
    growth, n = r.stdout.split()
    assert int(n) == n_batches * 50_000
    print(f"{n_batches * 50_000} rows: RSS growth {float(growth):.0f} MiB")
    return float(growth)


def test_streaming_bounded_memory_500k(tmp_path):
    assert _rss_growth_mib(tmp_path, 10) < 400  # MiB, fresh process


@pytest.mark.slow
def test_streaming_bounded_memory_2m(tmp_path):
    assert _rss_growth_mib(tmp_path, 40) < 400  # MiB, fresh process


def test_nside32_one_file_per_cell(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 200_000, 0, partition_nside=32)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    assert idx.num_rows == len(set(idx["cell"].to_pylist())) > 10_000
    assert sum(idx["n_rows"].to_pylist()) == 200_000


def test_nside16_and_row_groups_full(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_ROW_GROUP", 100)
    monkeypatch.setattr(writer, "MAX_ROWS_PER_FILE", 450)
    write_synthetic_dataset(tmp_path / "ds", 20_000, 0, partition_nside=2)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    assert sum(idx["n_rows"].to_pylist()) == 20_000
    per_cell = {}
    for row in idx.to_pylist():
        md = pq.ParquetFile(tmp_path / "ds" / row["path"]).metadata
        sizes = [md.row_group(g).num_rows for g in range(md.num_row_groups)]
        assert all(x == 100 for x in sizes[:-1]) and row["n_rows"] <= 400
        per_cell[row["cell"]] = per_cell.get(row["cell"], 0) + 1
    assert max(per_cell.values()) > 1  # rollover happened
    write_synthetic_dataset(tmp_path / "ds16", 5_000, 1, partition_nside=16)
    assert read_manifest(tmp_path / "ds16/manifest.json").partitioning.nside == 16


def test_row_groups_full_default(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 300_000, 0, partition_nside=1)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    for row in idx.to_pylist():
        md = pq.ParquetFile(tmp_path / "ds" / row["path"]).metadata
        assert all(md.row_group(g).num_rows == writer._ROW_GROUP
                   for g in range(md.num_row_groups - 1))
    assert idx.num_rows <= 12


def test_extra_stats_validation(tmp_path):
    t = synthetic_point_table(100, 0)
    for bad in ("ra", "z"):
        with pytest.raises(ValidationError, match="core"):
            _w(t, tmp_path / "x", extra_stats_columns=[bad])
    assert not (tmp_path / "x.tmp").exists()


def test_int64_stats_round_outwards(tmp_path):
    t = synthetic_point_table(50, 0)
    # 2**53+1 rounds down and 2**53+3 rounds up as float64: each breaks one bound
    t = t.append_column("lo", pa.array(np.full(50, 2**53 + 3, dtype=np.int64)))
    t = t.append_column("hi", pa.array(np.full(50, 2**53 + 1, dtype=np.int64)))
    _w(t, tmp_path / "ds", extra_stats_columns=["lo", "hi"], partition_nside=1)
    idx = read_partition_index(tmp_path / "ds/_index.parquet")
    assert min(idx["lo_min"].to_pylist()) <= 2**53 + 3
    assert max(idx["hi_max"].to_pylist()) >= 2**53 + 1


def test_overwrite_non_ouf3_manifest_message(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds/manifest.json").write_text('{"format_version": "2.0.0"}')
    with pytest.raises(FormatError, match="not an OUF 3.0 dataset"):
        _w(synthetic_point_table(10, 0), tmp_path / "ds", overwrite=True)


def test_overwrite_is_atomic_and_previous_untouched_on_failure(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 1_000, 0)
    before = read_manifest(tmp_path / "ds/manifest.json")
    with pytest.raises(FormatError):
        _w(synthetic_point_table(10, 0), tmp_path / "ds")  # no overwrite
    assert not (tmp_path / "ds.tmp").exists()

    def bad():
        yield synthetic_point_table(10, 0).to_batches()[0]
        raise RuntimeError("loader died")
    with pytest.raises(RuntimeError):
        _w(bad(), tmp_path / "ds", overwrite=True)
    assert read_manifest(tmp_path / "ds/manifest.json") == before
    assert not (tmp_path / "ds.tmp").exists()


def test_refuses_to_overwrite_non_dataset_dir(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds/precious.txt").write_text("x")
    with pytest.raises(FormatError, match="manifest"):
        _w(synthetic_point_table(10, 0), tmp_path / "ds", overwrite=True)
    assert (tmp_path / "ds/precious.txt").exists()


def test_rejects_unknown_z_type_and_missing_core(tmp_path):
    t = synthetic_point_table(100, 0)
    bad = t.set_column(t.schema.get_field_index("z_type"), "z_type", pa.array(["weird"] * 100))
    with pytest.raises(SchemaError, match="weird"):
        _w(bad, tmp_path / "a")
    with pytest.raises(SchemaError, match="dec"):
        _w(t.drop(["dec"]), tmp_path / "b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "a.tmp").exists()


def test_rejects_empty_and_schema_drift_and_bad_nside(tmp_path):
    t = synthetic_point_table(100, 0)
    with pytest.raises(ValidationError, match="no rows"):
        _w(iter(()), tmp_path / "e")
    with pytest.raises(ValidationError, match="no rows"):
        _w(t.slice(0, 0), tmp_path / "e2")
    b1 = t.to_batches()[0]
    b2 = pa.Table.from_batches([b1]).drop(["mag"]).to_batches()[0]
    with pytest.raises(SchemaError, match="schema"):
        _w(iter([b1, b2]), tmp_path / "d")
    with pytest.raises(ValidationError, match="partition_nside"):
        _w(t, tmp_path / "n", partition_nside=3)
    assert not (tmp_path / "d.tmp").exists()


def test_dictionary_and_plain_batches_mix(tmp_path):
    t = synthetic_point_table(100, 0)
    plain = t.to_batches()[0]
    dic = pa.Table.from_batches([plain]).set_column(
        t.schema.get_field_index("z_type"), "z_type",
        pa.array(["spec"] * 100).dictionary_encode()).to_batches()[0]
    m = _w(iter([plain, dic]), tmp_path / "ds")
    assert m.n_rows == 200 and pa.types.is_dictionary(m.schema.field("z_type").type)


def test_original_files_and_extensions(tmp_path):
    src = tmp_path / "cat.fits"
    src.write_bytes(b"abc")
    m = _w(synthetic_point_table(50, 0), tmp_path / "ds",
           original_files=[src, OriginalFile("other.fits", "ff", 7)],
           extensions={"coordinate": {"version": 1, "frame": "icrs"}})
    assert m.original_files[0].path == "cat.fits" and m.original_files[0].n_rows is None
    assert len(m.original_files[0].sha256) == 16
    assert m.original_files[1] == OriginalFile("other.fits", "ff", 7)
    assert m.extension("coordinate")["frame"] == "icrs"
    with pytest.raises(FormatError):
        _w(synthetic_point_table(50, 0), tmp_path / "bad", extensions={"pdf": {"version": 9}})
    assert not (tmp_path / "bad").exists()


def test_sightline_geometry(tmp_path):
    m = write_synthetic_sightlines(tmp_path / "sl", 300, 0)
    mm = read_manifest(tmp_path / "sl/manifest.json")
    assert mm.geometry == Geometry.SIGHTLINE and mm.n_rows == 300 == m.n_rows
    files = sorted((tmp_path / "sl/data").rglob("*.parquet"))
    t = pa.concat_tables([pq.read_table(f) for f in files])
    assert t.num_rows == 300 and t["delta"].type == pa.list_(pa.float32())
    assert "z_type" not in t.column_names
    t0 = synthetic_sightline_table(300, 0)
    with pytest.raises(SchemaError, match="wavelength"):
        write_dataset(t0.drop(["wavelength"]), tmp_path / "bad", geometry=Geometry.SIGHTLINE,
                      survey=SurveyMeta("lya", "lyman_alpha"), loader=L)


def test_validity_version_increments_on_overwrite(tmp_path):
    m1 = write_synthetic_dataset(tmp_path / "ds", 100, 0)
    m2 = write_synthetic_dataset(tmp_path / "ds", 100, 1, overwrite=True)
    assert m2.validity.version == 2 and (tmp_path / "ds/_history").exists()
    hist = list((tmp_path / "ds/_history").glob("*.manifest.json"))
    assert len(hist) == 1 and ":" not in hist[0].name
    assert read_manifest(hist[0]) == m1
    m3 = write_synthetic_dataset(tmp_path / "ds", 100, 2, overwrite=True)
    assert m3.validity.version == 3
    assert len(list((tmp_path / "ds/_history").glob("*.manifest.json"))) == 2


def _rows_by_file(root):
    idx = read_partition_index(root / "_index.parquet")
    return [(r["path"], pq.read_table(root / r["path"], columns=["source_row", "hpix32"]))
            for r in idx.to_pylist()]


def test_row_order_independent_of_batch_splits(tmp_path):
    t = synthetic_point_table(30_000, 3)
    for name, bs in (("a", 30_000), ("b", 2_500), ("c", 777)):
        _w(t, tmp_path / name, partition_nside=2, batch_size=bs)
    ref = _rows_by_file(tmp_path / "a")
    for name in ("b", "c"):
        got = _rows_by_file(tmp_path / name)
        assert [p for p, _ in got] == [p for p, _ in ref]
        assert all(x.equals(y) for (_, x), (_, y) in zip(got, ref, strict=True))
    for _, tab in ref:
        h, s = tab["hpix32"].to_numpy(), tab["source_row"].to_numpy()
        assert np.all((np.diff(h) > 0) | ((np.diff(h) == 0) & (np.diff(s) > 0)))


def test_row_order_with_many_stage_files(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "_STAGE_BUFFER_BYTES", 20_000)  # > 10 stage files per bucket
    t = synthetic_point_table(30_000, 4)
    _w(t, tmp_path / "a", partition_nside=2)
    monkeypatch.setattr(writer, "_STAGE_BUFFER_BYTES", 48 * 2**20)
    _w(t, tmp_path / "b", partition_nside=2)
    for (_, x), (_, y) in zip(_rows_by_file(tmp_path / "a"), _rows_by_file(tmp_path / "b"), strict=True):
        assert x.equals(y)


def test_caller_hpix32_is_ignored(tmp_path):
    t = synthetic_point_table(5_000, 0)
    good = _w(t, tmp_path / "g", partition_nside=4)
    if "hpix32" in t.schema.names:
        t = t.drop_columns(["hpix32"])
    t = t.append_column("hpix32", pa.array(np.zeros(t.num_rows, np.int32)))
    _w(t, tmp_path / "w", partition_nside=4)
    from oneuniverse.data import DatasetView
    a, b = DatasetView.from_path(tmp_path / "g"), DatasetView.from_path(tmp_path / "w")
    assert good.n_rows == b.n_rows
    assert a.index["cell"].to_pylist() == b.index["cell"].to_pylist()
    assert a.table(columns=["hpix32"])["hpix32"].equals(b.table(columns=["hpix32"])["hpix32"])


def test_user_iterator_oserror_propagates_unchanged(tmp_path):
    t = synthetic_point_table(4_000, 0)

    def gen(at):
        for i, b in enumerate(t.to_batches(max_chunksize=1_000)):
            if i == at:
                raise OSError("disk on fire")
            yield b
    for at in (0, 2):
        with pytest.raises(OSError, match="disk on fire") as ei:
            _w(gen(at), tmp_path / f"x{at}")
        assert not isinstance(ei.value, FormatError)
        assert not (tmp_path / f"x{at}").exists()


def test_dataset_uuid_stamped_in_every_file(tmp_path):
    m = write_synthetic_dataset(tmp_path / "ds", 20_000, 0, partition_nside=4)
    assert len(m.dataset_uuid) == 32
    for f in (tmp_path / "ds/data").rglob("*.parquet"):
        kv = pq.ParquetFile(f).metadata.metadata
        assert kv[b"oneuniverse.dataset_uuid"].decode() == m.dataset_uuid
    m2 = write_synthetic_dataset(tmp_path / "ds", 20_000, 0, partition_nside=4, overwrite=True)
    assert m2.dataset_uuid != m.dataset_uuid


def test_small_max_rows_per_file_still_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "MAX_ROWS_PER_FILE", 10)  # below one row group
    m = write_synthetic_dataset(tmp_path / "ds", 3_000, 0, partition_nside=1)
    assert m.n_rows == 3_000
