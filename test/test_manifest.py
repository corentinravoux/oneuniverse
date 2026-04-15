"""Tests for the OUF 2.0 typed Manifest dataclass and I/O."""
import json
from pathlib import Path

import pytest

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    FORMAT_VERSION,
    SCHEMA_VERSION,
    ColumnSpec,
    LoaderSpec,
    Manifest,
    ManifestValidationError,
    OriginalFileSpec,
    PartitionSpec,
    PartitionStats,
    PartitioningSpec,
    read_manifest,
    write_manifest,
)


def _sample_manifest() -> Manifest:
    return Manifest(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=SCHEMA_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="test",
        survey_type="spectroscopic",
        created_utc="2026-04-15T12:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="test.fits",
            sha256="deadbeefcafebabe",
            n_rows=100,
            size_bytes=4096,
            format="fits",
        )],
        partitions=[PartitionSpec(
            name="part_0000.parquet",
            n_rows=100,
            sha256="abad1deaabad1dea",
            size_bytes=2048,
            stats=PartitionStats(
                ra_min=0.0, ra_max=10.0,
                dec_min=-5.0, dec_max=5.0,
                z_min=0.1, z_max=0.5,
            ),
        )],
        partitioning=None,
        schema=[
            ColumnSpec(name="ra", dtype="float64", unit="deg", required=True),
            ColumnSpec(name="dec", dtype="float64", unit="deg", required=True),
        ],
        conversion_kwargs={},
        loader=LoaderSpec(name="test_loader", version="0.1.0"),
    )


def test_manifest_roundtrip(tmp_path: Path):
    m = _sample_manifest()
    target = tmp_path / "manifest.json"
    write_manifest(target, m)
    loaded = read_manifest(target)
    assert loaded == m


def test_manifest_rejects_wrong_format_version(tmp_path: Path):
    m = _sample_manifest()
    target = tmp_path / "manifest.json"
    write_manifest(target, m)
    raw = json.loads(target.read_text())
    raw["oneuniverse_format_version"] = "1.0.0"
    target.write_text(json.dumps(raw))
    with pytest.raises(ManifestValidationError, match="format_version"):
        read_manifest(target)


def test_manifest_rejects_missing_required_keys(tmp_path: Path):
    target = tmp_path / "manifest.json"
    target.write_text('{"geometry": "point"}')
    with pytest.raises(ManifestValidationError):
        read_manifest(target)


def test_manifest_rejects_bad_geometry(tmp_path: Path):
    m = _sample_manifest()
    target = tmp_path / "manifest.json"
    write_manifest(target, m)
    raw = json.loads(target.read_text())
    raw["geometry"] = "volumetric"
    target.write_text(json.dumps(raw))
    with pytest.raises(ManifestValidationError, match="geometry"):
        read_manifest(target)


def test_manifest_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ManifestValidationError, match="not found"):
        read_manifest(tmp_path / "no_such.json")


def test_manifest_rejects_non_object_top(tmp_path: Path):
    target = tmp_path / "manifest.json"
    target.write_text('["not", "an", "object"]')
    with pytest.raises(ManifestValidationError, match="object"):
        read_manifest(target)


def test_manifest_rejects_invalid_json(tmp_path: Path):
    target = tmp_path / "manifest.json"
    target.write_text("{not json")
    with pytest.raises(ManifestValidationError, match="JSON"):
        read_manifest(target)


def test_manifest_write_is_atomic(tmp_path: Path):
    m = _sample_manifest()
    target = tmp_path / "manifest.json"
    write_manifest(target, m)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["manifest.json"]


def test_manifest_n_rows_and_n_partitions(tmp_path: Path):
    m = Manifest(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=SCHEMA_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="test",
        survey_type="t",
        created_utc="2026-04-15T12:00:00+00:00",
        original_files=[],
        partitions=[
            PartitionSpec(name="p0", n_rows=10, sha256="a" * 16, size_bytes=1),
            PartitionSpec(name="p1", n_rows=20, sha256="b" * 16, size_bytes=2),
        ],
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="l", version="0.0.0"),
    )
    assert m.n_rows == 30
    assert m.n_partitions == 2


def test_manifest_with_partitioning(tmp_path: Path):
    m = _sample_manifest()
    # Replace partitioning with an explicit spec.
    m2 = Manifest(
        **{**m.__dict__, "partitioning": PartitioningSpec(
            scheme="healpix32", column="_healpix32", extra={"nside": 32},
        )}
    )
    target = tmp_path / "manifest.json"
    write_manifest(target, m2)
    loaded = read_manifest(target)
    assert loaded.partitioning is not None
    assert loaded.partitioning.scheme == "healpix32"
    assert loaded.partitioning.column == "_healpix32"
    assert loaded.partitioning.extra["nside"] == 32
