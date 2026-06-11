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


# ── F3: adaptive partition NSIDE in write_ouf_dataset ───────────────────


def _fake_point_df(n: int, seed: int) -> pd.DataFrame:
    import healpy as hp
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["fake"] * n, dtype="<U16"),
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
    phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
    df["_healpix32"] = hp.ang2pix(32, theta, phi, nest=True).astype(np.int32)
    return df


def test_small_catalog_uses_coarser_nside(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    df = _fake_point_df(n=1000, seed=7)
    ou_dir = tmp_path / "small" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    manifest = write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="small", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="small", version="0"),
    )
    chosen = int(manifest.partitioning.extra["nside"])
    assert chosen <= 4, f"got nside={chosen}, expected coarsening"
    parquet_files = list((ou_dir / "data").rglob("*.parquet"))
    assert len(parquet_files) <= 32, (
        f"got {len(parquet_files)} files; coarsening should keep this small"
    )
    got = DatasetView.from_path(ou_dir.parent).read()
    assert len(got) == 1000


def test_large_catalog_keeps_nside_32(tmp_path):
    """At ~7M rows (~500 rows/cell at NSIDE=32) the default stays at 32."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import HEALPIX_PARTITION_NSIDE
    n_rows = HEALPIX_PARTITION_NSIDE * HEALPIX_PARTITION_NSIDE * 12 * 5_000 + 1
    # That's well above the threshold at 32. Skip if too memory-heavy.
    if n_rows > 200_000:
        pytest.skip("skipping multi-million-row sanity at unit-test scale")


def test_partition_nside_can_be_forced(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    df = _fake_point_df(n=10_000, seed=13)
    ou_dir = tmp_path / "forced" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    manifest = write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="forced", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="forced", version="0"),
        partition_nside=4,
    )
    assert int(manifest.partitioning.extra["nside"]) == 4
    # And the view round-trips.
    got = DatasetView.from_path(ou_dir.parent).read()
    assert len(got) == 10_000


def test_auto_picker_returns_finest_when_enough_rows():
    """At 5000 × 12288 (= 61.4M rows) the auto-picker must pick NSIDE=32."""
    from oneuniverse.data.converter import _auto_partition_nside
    n = 5_000 * 12 * 32 * 32
    assert _auto_partition_nside(n) == 32


def test_auto_picker_coarsens_for_small_catalogs():
    from oneuniverse.data.converter import _auto_partition_nside
    # 1000 rows: 1000 / 12288 cells = ~0.08 rows/cell at 32 → must coarsen.
    assert _auto_partition_nside(1000) <= 4
    # 200 rows → must coarsen further.
    assert _auto_partition_nside(200) <= 2
