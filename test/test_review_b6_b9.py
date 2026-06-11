"""Regressions for review items B6-B9.

B6 parquet linkback · B7 area-uniform randoms (NEST sub-pixels, no escapers)
B8 relative covariance paths (a moved MeasurementSet still resolves)
B9 honest device stat when GPU is requested on the wrap-in-place branch.
"""
import shutil
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.covariance import CovarianceHandle, CovariancePlan
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.nz import Nz
from oneuniverse.measure.randoms import generate_randoms
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.window import Window

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


# ── B6: parquet originals support linkback ──────────────────────────────────
def test_b6_parquet_linkback(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from oneuniverse.data.converter import (fetch_original_columns,
                                            write_ouf_dataset)
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec
    n = 50
    rng = np.random.default_rng(0)
    ra = rng.uniform(10, 20, n); dec = rng.uniform(0, 5, n)
    orig = tmp_path / "survey" / "orig.parquet"
    orig.parent.mkdir(parents=True)
    pq.write_table(pa.table({"ra": ra, "dec": dec,
                             "secret": np.arange(n) * 2.0}), orig)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": rng.uniform(0.1, 1, n),
        "z_type": ["spec"] * n, "z_err": [1e-4] * n,
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True,
                                 lonlat=True).astype("i4")})
    write_ouf_dataset(df=df, out_dir=tmp_path / "survey" / "oneuniverse",
                      survey_name="pq", survey_type="spectroscopic",
                      geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name="pq", version="0"),
                      original_paths=[orig], original_format="parquet")
    got = fetch_original_columns(tmp_path / "survey", ["secret"],
                                 row_indices=np.array([3, 7]))
    np.testing.assert_allclose(got["secret"].to_numpy(), [6.0, 14.0])


# ── B7: randoms are strictly in-window and pole-safe (no snap hack) ─────────
def test_b7_randoms_in_window_including_polar_pixels():
    nside = 16
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix)
    # cover the 40 NEST pixels nearest each pole + an equatorial band
    th, _ = hp.pix2ang(nside, np.arange(npix), nest=True)
    mask[np.argsort(th)[:40]] = 1.0           # north polar cap pixels
    mask[np.argsort(-th)[:40]] = 1.0          # south polar cap pixels
    win = Window(nside=nside, mask=mask)
    nz = Nz(np.linspace(0, 1, 5), np.ones(4), "spec_hist")
    rnd, _ = generate_randoms(win, nz, n_randoms=5000, seed=3)
    # NEST children are strictly nested: every random is inside the window
    assert win.contains(rnd["ra"].to_numpy(), rnd["dec"].to_numpy()).all()
    assert (np.abs(rnd["dec"]) > 60).mean() > 0.9   # really polar


# ── B8: a moved MeasurementSet still resolves its covariance ────────────────
def test_b8_covariance_path_survives_directory_move(tmp_path):
    n = 10
    set_dir = tmp_path / "ms"
    set_dir.mkdir()
    cov = np.diag(np.ones(n)); np.save(set_dir / "cov.npy", cov)
    plan = CovariancePlan(kind="external",
                          handle=CovarianceHandle("c", str(set_dir / "cov.npy"),
                                                  n))
    ps = PointSet(catalog=pd.DataFrame({"ra": np.zeros(n), "dec": np.zeros(n),
                                        "z": np.full(n, .5)}),
                  region_map=np.zeros(n, dtype=np.int64),
                  metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                           length_unit="deg", nside_region=8),
                  provenance=Provenance(dataset_ids=("x",)), covariance=plan)
    ms = MeasurementSet({"g": ps}, MeasurementSpec(
        ("g",), (("g", "g"),), "hubble", "sn"), ps.metadata)
    ms.to_dir(set_dir)
    moved = tmp_path / "elsewhere"
    shutil.move(str(set_dir), moved)                  # relocate the whole set
    back = MeasurementSet.from_dir(moved)
    mat = back.products["g"].covariance.handle.matrix()   # resolves post-move
    np.testing.assert_allclose(np.diag(mat), 1.0)


# ── B9: GPU request on a wrap-in-place store reports cpu(reference) ─────────
def test_b9_device_stat_honest_on_reference_branch(tmp_path):
    from oneuniverse.simulation.cosmology import CosmologySpec
    from oneuniverse.simulation.linear import generate_linear_sim
    from oneuniverse.simulation.linear.pack import write_packed_native
    from oneuniverse.simulation.oufsim import SimStore
    from oneuniverse.simulation.packed.converter import PackedSimConverter
    from oneuniverse.simulation.selectors import Cube
    C = CosmologySpec(omega_m=.31, omega_b=.048, h=.67, n_s=.96, sigma8=.81,
                      t_cmb=2.7255)
    lin = generate_linear_sim(tmp_path / "n", C, box_size=150.0, n_grid=16,
                              redshifts=(0.0,), seed=1, with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "p", particle_chunk_nside=2)
    ref = PackedSimConverter().convert(pk, tmp_path / "r", sim_name="e",
                                       projection="reference")
    s = SimStore(ref)
    s.read_box("snapshots", 0.0, Cube(0, 75, 0, 75, 0, 75), device="gpu")
    dev = s.last_read_stats["device"]
    assert dev in ("cpu", "cpu(reference)")
    assert dev != "gpu"                  # never claims GPU on the memmap branch
