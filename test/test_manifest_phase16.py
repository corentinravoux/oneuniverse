"""Phase 16 T6 — Manifest carries CoordinateSpec / SpectrumSpec /
observed_z_types and round-trips OUF 2.2 ↔ 2.1.
"""
import json

import pytest

from oneuniverse.data.coordinate_spec import CoordinateSpec
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
from oneuniverse.data.spectrum_spec import SpectrumSpec


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version="2.2.0",
        geometry=DataGeometry.POINT,
        survey_name="fixture",
        survey_type="spectroscopic",
        created_utc="2026-05-28T00:00:00+00:00",
        original_files=[
            OriginalFileSpec(
                path="raw.fits",
                sha256="0123456789abcdef",
                n_rows=10,
                size_bytes=4096,
                format="fits",
            ),
        ],
        partitions=[
            PartitionSpec(
                name="data/part_0000.parquet",
                n_rows=10,
                sha256="fedcba9876543210",
                size_bytes=2048,
                stats=PartitionStats(),
            ),
        ],
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.4.0"


def test_manifest_carries_coordinate_spec(tmp_path):
    m = _minimal_manifest(
        coordinate=CoordinateSpec(
            frame="icrs", epoch=2016.0, proper_motion_available=True,
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.coordinate == m.coordinate


def test_manifest_carries_spectrum_spec(tmp_path):
    m = _minimal_manifest(
        geometry=DataGeometry.SIGHTLINE,
        spectrum=SpectrumSpec(
            wavelength_convention="vacuum",
            log_binned=True,
            rest_frame_corrected=False,
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.spectrum == m.spectrum


def test_manifest_carries_observed_z_types(tmp_path):
    m = _minimal_manifest(observed_z_types=("spec", "phot"))
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert tuple(read.observed_z_types) == ("spec", "phot")


def test_reads_phase15_2_1_manifest_with_compat_defaults(tmp_path):
    """A 2.1.0 manifest written before Phase 16 must still parse."""
    payload = {
        "oneuniverse_format_version": "2.1.0",
        "oneuniverse_schema_version": "2.1.0",
        "geometry": "point",
        "survey_name": "legacy",
        "survey_type": "spectroscopic",
        "created_utc": "2026-04-15T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
        }],
        "partitioning": None,
        "schema": [],
        "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    m = read_manifest(path)
    assert m.coordinate is None
    assert m.spectrum is None
    assert tuple(m.observed_z_types) == ()


def test_unknown_format_version_still_rejected(tmp_path):
    from oneuniverse.data.manifest import ManifestValidationError

    payload = {
        "oneuniverse_format_version": "3.0.0",
        "oneuniverse_schema_version": "3.0.0",
        "geometry": "point",
        "survey_name": "future",
        "survey_type": "spectroscopic",
        "created_utc": "2030-01-01T00:00:00+00:00",
        "original_files": [],
        "partitions": [],
        "partitioning": None,
        "schema": [],
        "conversion_kwargs": {},
        "loader": {"name": "future_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError):
        read_manifest(path)
