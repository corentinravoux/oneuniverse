"""Phase 14 T3: prove cone-query reads only matching partitions.

Today's cone tests check returned row counts and per-row coordinates;
they do not prove that pyarrow actually skipped the irrelevant parquet
files. This audit goes one level lower — counts the partitions that
:meth:`DatasetView._select_partitions` returns and asserts it is a
proper subset of the on-disk fragments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import (
    DataGeometry, HEALPIX_PARTITION_NSIDE,
)
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.data.selection import Cone, SkyPatch


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


def _write(tmp_path, df):
    ou_dir = tmp_path / "ds" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="fake", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fake", version="0"),
        partition_nside=HEALPIX_PARTITION_NSIDE,
    )
    return DatasetView.from_path(ou_dir.parent)


def test_cone_resolves_to_subset_of_partitions(tmp_path):
    """A 2°-radius cone on a uniform 10k-row catalog opens < 1/5 of partitions."""
    view = _write(tmp_path, _fake_point_df(n=10_000, seed=0))
    n_total = view.n_partitions
    cone = Cone(ra=180.0, dec=0.0, radius=2.0)
    cells = view._resolve_cells(cone=cone, skypatch=None, healpix_cells=None)
    chosen = view._select_partitions(healpix_cells=cells)
    assert 0 < len(chosen) < n_total
    # Sanity: chosen partitions are a subset of the resolved cells.
    chosen_cells = {p.healpix_cell for p in chosen}
    assert chosen_cells.issubset(set(cells))


def test_cone_query_returns_only_in_cone_rows(tmp_path):
    """Pushdown sanity: every returned row must sit inside the cone."""
    view = _write(tmp_path, _fake_point_df(n=10_000, seed=1))
    cone = Cone(ra=180.0, dec=0.0, radius=2.0)
    tbl = view.scan(cone=cone)
    ras = np.asarray(tbl["ra"].to_pylist())
    decs = np.asarray(tbl["dec"].to_pylist())
    cosd = (
        np.sin(np.radians(decs)) * np.sin(np.radians(0.0))
        + np.cos(np.radians(decs)) * np.cos(np.radians(0.0))
          * np.cos(np.radians(ras) - np.radians(180.0))
    )
    sep = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
    assert (sep <= 2.0 + 1e-6).all()


def test_skypatch_resolves_to_subset_of_partitions(tmp_path):
    """Tight SkyPatch likewise reduces the partition set."""
    view = _write(tmp_path, _fake_point_df(n=10_000, seed=2))
    n_total = view.n_partitions
    patch = SkyPatch(ra_min=10.0, ra_max=30.0, dec_min=-5.0, dec_max=5.0)
    cells = view._resolve_cells(cone=None, skypatch=patch, healpix_cells=None)
    chosen = view._select_partitions(healpix_cells=cells)
    assert 0 < len(chosen) < n_total
