import json

import pyarrow as pa
import pytest

from oneuniverse._core.errors import FormatError
from oneuniverse.data.manifest import (FORMAT_VERSION, LoaderInfo, Manifest,
                                       OriginalFile, Partitioning, SurveyMeta,
                                       Validity, read_manifest,
                                       read_partition_index, write_manifest,
                                       write_partition_index)
from oneuniverse.data.schema import Geometry, core_schema


def _manifest(**kw) -> Manifest:
    base = dict(
        format_version=FORMAT_VERSION, geometry=Geometry.POINT,
        survey=SurveyMeta("s", "spectroscopic"), loader=LoaderInfo("l", "1", {"a": 1}),
        created_utc="2026-11-12T00:00:00Z", schema=core_schema(Geometry.POINT),
        partitioning=Partitioning(32),
        original_files=(OriginalFile("f.fits", "ab" * 32, None),), n_rows=10,
        validity=Validity("2026-11-12T00:00:00Z", None, 1),
        extensions={"coordinate": {"version": 1, "frame": "icrs"}})
    base.update(kw)
    return Manifest(**base)


def test_manifest_round_trip_preserves_arrow_schema(tmp_path):
    m = _manifest(); write_manifest(tmp_path / "manifest.json", m); back = read_manifest(tmp_path / "manifest.json")
    assert back.schema.equals(m.schema) and back == m
    assert back != _manifest(n_rows=11)


def test_manifest_json_is_human_readable(tmp_path):
    write_manifest(tmp_path / "m.json", _manifest()); d = json.loads((tmp_path / "m.json").read_text())
    assert d["format_version"] == "3.0.0" and d["schema"]["columns"][0] == {"name": "ra", "type": "double"} and "ipc_base64" in d["schema"]


def test_read_manifest_rejects_other_major(tmp_path):
    d = _manifest().to_dict(); d["format_version"] = "2.6.0"; (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="2.6.0"): read_manifest(tmp_path / "m.json")


def test_read_manifest_rejects_missing_key(tmp_path):
    d = _manifest().to_dict(); del d["partitioning"]; (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="partitioning"): read_manifest(tmp_path / "m.json")


def test_extension_version_required():
    with pytest.raises(FormatError, match="version"): _manifest(extensions={"coordinate": {"frame": "icrs"}})
    with pytest.raises(FormatError, match="version"): _manifest(extensions={"coordinate": {"version": 9}})
    _manifest(extensions={"custom": {"version": 7}})


def test_manifest_unhashable_and_extension_lookup():
    m = _manifest()
    with pytest.raises(TypeError): hash(m)
    assert m.extension("coordinate")["frame"] == "icrs" and m.extension("nope") is None


def test_partition_index_round_trip(tmp_path):
    t = pa.table({"path": ["data/cell=1/part-0.parquet"], "cell": pa.array([1], pa.int32()), "n_rows": [10], "ra_min": [0.0], "ra_max": [1.0], "dec_min": [0.0], "dec_max": [1.0], "z_min": [0.1], "z_max": [0.2]})
    write_partition_index(tmp_path / "_index.parquet", t); assert read_partition_index(tmp_path / "_index.parquet").equals(t)
    with pytest.raises(FormatError): write_partition_index(tmp_path / "x.parquet", t.drop(["cell"]))
    assert not (tmp_path / "x.parquet").exists()
