"""H2 fix — MeasurementSet round-trips to/from disk (the P1→P2 handoff)."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure import (MeasurementSet, build_cosmic_shear, build_lya,
                                 build_galaxy_clustering, build_map_cross)
from oneuniverse.measure.fieldmap import fieldmap_from_healpix

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import (synthetic_healpix_map,  # noqa: E402
    synthetic_point_view, synthetic_shear_view, synthetic_sightline_view)


def _fkp():
    return FKPWeight(nbar=lambda z: np.full_like(z, 1e-3), P0=1e4)


def test_clustering_roundtrip_exact(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=1)
    ms = build_galaxy_clustering(view, tracer="gal", z_range=(0.2, 0.9),
                                 weights=[_fkp(), ColumnWeight("weight_comp")],
                                 nz_edges=np.linspace(0, 1.2, 25),
                                 randoms="generate", n_randoms=6000, seed=2)
    ms.to_dir(tmp_path / "ms")
    back = MeasurementSet.from_dir(tmp_path / "ms")
    a, b = ms.products["gal"], back.products["gal"]
    np.testing.assert_array_equal(a.region_map, b.region_map)
    np.testing.assert_allclose(a.catalog["weight"].to_numpy(),
                               b.catalog["weight"].to_numpy())
    assert len(a.randoms) == len(b.randoms)
    np.testing.assert_allclose(a.nz.counts, b.nz.counts)
    np.testing.assert_allclose(a.window.mask, b.window.mask)
    assert back.spec.statistic == "pk_multipole"
    assert back.metadata.nside_region == ms.metadata.nside_region
    back.check_invariants()


def test_cosmic_shear_roundtrip_photoz_and_tomo(tmp_path):
    view = synthetic_shear_view(tmp_path, n=2000, seed=3, with_pdf=True,
                                n_tomo=2)
    ms = build_cosmic_shear(view, z_grid=np.linspace(0, 2, 41), nside_region=4)
    ms.to_dir(tmp_path / "wl")
    back = MeasurementSet.from_dir(tmp_path / "wl")
    a, b = ms.products["src"], back.products["src"]
    assert isinstance(b.nz, dict) and set(b.nz) == set(a.nz)
    assert b.photoz is not None and b.photoz.values.shape[0] == len(b.catalog)
    np.testing.assert_array_equal(a.tomo_bin, b.tomo_bin)
    back.check_invariants()


def test_sightline_roundtrip_ragged(tmp_path):
    view = synthetic_sightline_view(tmp_path, n_los=8, n_pix=12, seed=2)
    ms = build_lya(view, nside_region=16)
    ms.to_dir(tmp_path / "lya")
    back = MeasurementSet.from_dir(tmp_path / "lya")
    a, b = ms.products["lya"], back.products["lya"]
    assert b.n_sightlines == a.n_sightlines
    for i in range(a.n_sightlines):
        np.testing.assert_allclose(a.delta[i], b.delta[i])


def test_fieldmap_roundtrip(tmp_path):
    gview = synthetic_point_view(tmp_path, n=2000, seed=3, name="g")
    vals, mask = synthetic_healpix_map(nside=16, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=16, dataset_id="cmbk")
    ms = build_map_cross(gview, fm, nside_region=4, z_range=(0.1, 1.0))
    ms.to_dir(tmp_path / "mx")
    back = MeasurementSet.from_dir(tmp_path / "mx")
    assert set(back.products) == {"gal", "kappa"}
    bm = back.products["kappa"]
    np.testing.assert_allclose(fm.values, bm.values)
    np.testing.assert_array_equal(fm.mask, bm.mask)
    back.check_invariants()
