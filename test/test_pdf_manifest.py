"""PdfSpec round-trip through Manifest read/write."""
from __future__ import annotations

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec, read_manifest,
    write_manifest,
)
from oneuniverse.data.pdf import PdfSpec


def _make_minimal_manifest(pdf_spec):
    return Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="fake",
        survey_type="photometric",
        created_utc="2026-04-23T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="x.fits", sha256="abc", n_rows=1, size_bytes=1, format="fits"
        )],
        partitions=[],
        partitioning=None,
        schema=[ColumnSpec(name="z", dtype="f4")],
        conversion_kwargs={},
        loader=LoaderSpec(name="fake", version="0"),
        pdf_spec=pdf_spec,
    )


def test_manifest_roundtrip_with_pdf_spec(tmp_path):
    spec = PdfSpec(
        parameterisation="interp", n_components=3, grid=[0.0, 0.5, 1.0],
        grid_kind="z",
    )
    m = _make_minimal_manifest(spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    back = read_manifest(path)
    assert back.pdf_spec == spec


def test_manifest_pdf_spec_is_none_by_default(tmp_path):
    m = _make_minimal_manifest(None)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    back = read_manifest(path)
    assert back.pdf_spec is None
