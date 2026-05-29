"""Phase 18 T6 — Manifest carries tomographic_nz + classification_pdf
and round-trips OUF 2.4 ↔ 2.3.
"""
import json

import pytest

from oneuniverse.data.classification_pdf import ClassificationPdfSpec
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
from oneuniverse.data.tomographic_nz import TomographicNzSpec


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="fixture", survey_type="photometric",
        created_utc="2026-05-29T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="raw.fits", sha256="0123456789abcdef",
            n_rows=10, size_bytes=4096, format="fits",
        )],
        partitions=[PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=10, sha256="fedcba9876543210", size_bytes=2048,
            stats=PartitionStats(),
        )],
        partitioning=None, schema=[], conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.4.0"


def test_manifest_carries_tomographic_nz(tmp_path):
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6)],
        grid=[0.0, 0.5, 1.0],
        values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    m = _minimal_manifest(tomographic_nz=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.tomographic_nz == spec


def test_manifest_carries_classification_pdf(tmp_path):
    spec = ClassificationPdfSpec(classes=("galaxy", "qso", "star"))
    m = _minimal_manifest(classification_pdf=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.classification_pdf == spec


def test_reads_2_3_manifest_with_compat_defaults(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.3.0",
        "oneuniverse_schema_version": "2.3.0",
        "geometry": "point",
        "survey_name": "legacy", "survey_type": "photometric",
        "created_utc": "2026-05-28T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
        }],
        "partitioning": None, "schema": [], "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    read = read_manifest(path)
    assert read.tomographic_nz is None
    assert read.classification_pdf is None
