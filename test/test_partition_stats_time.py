import pytest

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest, read_manifest,
)
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


_VALID_T0 = "2026-01-01T00:00:00+00:00"


def _minimal(tmp_path, stats, temporal=None, validity=None):
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn",
        survey_type="transient",
        created_utc=_VALID_T0,
        original_files=[OriginalFileSpec(
            path="src.csv", sha256="0" * 16, n_rows=None,
            size_bytes=1, format="csv",
        )],
        partitions=[PartitionSpec(
            name="part_0000.parquet", n_rows=10,
            sha256="f" * 16, size_bytes=100, stats=stats,
        )],
        partitioning=None,
        schema=[ColumnSpec(name="t_obs", dtype="float64")],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=temporal,
        validity=(validity or DatasetValidity(valid_from_utc=_VALID_T0)),
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, m)
    return out


def test_partition_stats_accepts_time_fields():
    s = PartitionStats(t_min=58000.0, t_max=60000.0)
    assert s.t_min == 58000.0 and s.t_max == 60000.0


def test_partition_stats_time_defaults_none():
    s = PartitionStats()
    assert s.t_min is None and s.t_max is None


def test_manifest_roundtrip_preserves_time_stats(tmp_path):
    stats = PartitionStats(ra_min=0, ra_max=10, t_min=58000.0, t_max=60000.0)
    m = read_manifest(_minimal(tmp_path, stats))
    assert m.partitions[0].stats.t_min == 58000.0
    assert m.partitions[0].stats.t_max == 60000.0


def test_manifest_roundtrip_preserves_temporal_spec(tmp_path):
    spec = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=3.0)
    m = read_manifest(_minimal(tmp_path, PartitionStats(), temporal=spec))
    assert m.temporal == spec


def test_manifest_roundtrip_preserves_validity(tmp_path):
    v = DatasetValidity(valid_from_utc=_VALID_T0, version="dr17",
                        supersedes=("eboss_qso_dr16",))
    m = read_manifest(_minimal(tmp_path, PartitionStats(), validity=v))
    assert m.validity == v


def test_manifest_20_without_validity_is_defaulted(tmp_path):
    """A 2.0.x manifest on disk (no validity block) reads back with a
    default-filled validity. Forward-compatibility promise."""
    import json
    ok = _minimal(tmp_path, PartitionStats())
    raw = json.loads(ok.read_text())
    raw["oneuniverse_format_version"] = "2.0.0"
    raw.pop("validity", None)
    raw.pop("temporal", None)
    ok.write_text(json.dumps(raw))
    m = read_manifest(ok)
    assert m.validity is not None
    assert m.validity.valid_from_utc.startswith("2026-01-01")
    assert m.validity.valid_to_utc is None
    assert m.validity.version == "1.0"


def test_manifest_format_version_3x_rejected(tmp_path):
    import json
    ok = _minimal(tmp_path, PartitionStats())
    raw = json.loads(ok.read_text())
    raw["oneuniverse_format_version"] = "3.0.0"
    ok.write_text(json.dumps(raw))
    with pytest.raises(Exception):
        read_manifest(ok)
