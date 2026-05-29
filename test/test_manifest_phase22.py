"""Phase 22 T3 — Manifest carries cube / gwskymap and bumps to OUF 2.5."""
import json

import pytest

from oneuniverse.data.cube_spec import CubeSpec
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.gwskymap_spec import GwSkymapSpec
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


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.CUBE,
        survey_name="fixture", survey_type="ifu",
        created_utc="2026-05-29T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="raw.fits", sha256="0123456789abcdef",
            n_rows=1, size_bytes=100, format="fits",
        )],
        partitions=[PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=1, sha256="fedcba9876543210", size_bytes=50,
            stats=PartitionStats(),
        )],
        partitioning=None, schema=[], conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.5.0"


def test_manifest_carries_cube_spec(tmp_path):
    spec = CubeSpec(
        axes=("ra", "dec", "wavelength"),
        axis_units=("deg", "deg", "angstrom"),
        wavelength_convention="vacuum",
    )
    m = _minimal_manifest(cube=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.cube == spec


def test_manifest_carries_gwskymap_spec(tmp_path):
    spec = GwSkymapSpec(map_nside=32)
    m = _minimal_manifest(
        geometry=DataGeometry.GW_SKYMAP, gwskymap=spec,
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.gwskymap == spec


def test_reads_2_4_manifest_with_compat_defaults(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.4.0",
        "oneuniverse_schema_version": "2.4.0",
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
    assert read.cube is None
    assert read.gwskymap is None
