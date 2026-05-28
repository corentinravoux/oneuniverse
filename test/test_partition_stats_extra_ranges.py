"""Phase 17 T4 — PartitionStats.extra_ranges round-trips."""
import json

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    FORMAT_VERSION,
    LoaderSpec,
    Manifest,
    OriginalFileSpec,
    PartitionSpec,
    PartitionStats,
    read_manifest,
    write_manifest,
)


def _minimal(partitions):
    return Manifest(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="fixture",
        survey_type="spectroscopic",
        created_utc="2026-05-28T00:00:00+00:00",
        original_files=[
            OriginalFileSpec(
                path="raw.fits", sha256="0123456789abcdef",
                n_rows=1, size_bytes=100, format="fits",
            ),
        ],
        partitions=partitions,
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )


def test_extra_ranges_default_empty():
    stats = PartitionStats()
    assert stats.extra_ranges == {}


def test_extra_ranges_in_manifest_roundtrip(tmp_path):
    parts = [
        PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=1, sha256="fedcba9876543210", size_bytes=50,
            stats=PartitionStats(
                ra_min=0.0, ra_max=1.0,
                dec_min=-1.0, dec_max=1.0,
                z_min=0.1, z_max=0.5,
                extra_ranges={"snr": (10.0, 100.0), "ebv": (0.0, 0.05)},
            ),
        ),
    ]
    m = _minimal(parts)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    s = read.partitions[0].stats
    assert s.extra_ranges == {
        "snr": (10.0, 100.0), "ebv": (0.0, 0.05),
    }


def test_old_manifest_without_extra_ranges_parses(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.2.0",
        "oneuniverse_schema_version": "2.2.0",
        "geometry": "point",
        "survey_name": "legacy", "survey_type": "spectroscopic",
        "created_utc": "2026-05-28T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
            "stats": {"ra_min": 0.0, "ra_max": 1.0},
        }],
        "partitioning": None, "schema": [], "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    read = read_manifest(path)
    assert read.partitions[0].stats.extra_ranges == {}
