"""F3 + D5: adaptive partition NSIDE + manifest-driven cone resolution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    LoaderSpec, Manifest, PartitioningSpec,
)
from oneuniverse.data.selection import Cone


# ── D5: cone resolution must read NSIDE from manifest ───────────────────


def _make_view_with_partitioning_nside(nside: int) -> DatasetView:
    """Build a minimal DatasetView whose manifest claims partitioning NSIDE."""
    manifest = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="fake",
        survey_type="spectroscopic",
        created_utc="2026-05-22T00:00:00+00:00",
        original_files=[],
        partitions=[],
        partitioning=PartitioningSpec(
            scheme="healpix",
            column="_healpix32",
            extra={"nside": nside, "nest": True},
        ),
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fake", version="0"),
    )
    # ou_dir does not need to exist — _resolve_cells doesn't touch the disk.
    from pathlib import Path
    return DatasetView(ou_dir=Path("/tmp/none"), manifest=manifest)


def test_resolve_cells_uses_manifest_nside_8():
    """A manifest claiming NSIDE=8 must drive cone-cell resolution at 8, not 32."""
    import healpy as hp
    view = _make_view_with_partitioning_nside(nside=8)
    cone = Cone(ra=180.0, dec=0.0, radius=10.0)
    cells = view._resolve_cells(cone=cone, skypatch=None, healpix_cells=None)
    expected = sorted(int(c) for c in cone.healpix_cells(nside=8, nest=True))
    assert cells == expected
    npix8 = hp.nside2npix(8)
    assert all(0 <= c < npix8 for c in cells)


def test_resolve_cells_uses_manifest_nside_4():
    view = _make_view_with_partitioning_nside(nside=4)
    cone = Cone(ra=42.0, dec=-20.0, radius=15.0)
    cells = view._resolve_cells(cone=cone, skypatch=None, healpix_cells=None)
    expected = sorted(int(c) for c in cone.healpix_cells(nside=4, nest=True))
    assert cells == expected


def test_resolve_cells_default_nside_when_manifest_no_partitioning():
    """No partitioning block → fall back to the global default (32)."""
    from oneuniverse.data.format_spec import HEALPIX_PARTITION_NSIDE
    manifest = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="fake",
        survey_type="spectroscopic",
        created_utc="2026-05-22T00:00:00+00:00",
        original_files=[],
        partitions=[],
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fake", version="0"),
    )
    from pathlib import Path
    view = DatasetView(ou_dir=Path("/tmp/none"), manifest=manifest)
    cone = Cone(ra=10.0, dec=10.0, radius=5.0)
    cells = view._resolve_cells(cone=cone, skypatch=None, healpix_cells=None)
    expected = sorted(
        int(c) for c in cone.healpix_cells(
            nside=HEALPIX_PARTITION_NSIDE, nest=True,
        )
    )
    assert cells == expected
