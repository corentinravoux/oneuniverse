"""measure T8 — galaxy-clustering MeasurementSet end-to-end."""
import sys
from pathlib import Path

import numpy as np
import pytest

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure import build_galaxy_clustering
from oneuniverse.measure.measurement_set import MeasurementSet

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def _fkp():
    # FKPWeight.nbar is a callable z -> n̄(z); synthetic n̄ is constant 1e-3
    return FKPWeight(nbar=lambda z: np.full_like(z, 1e-3), P0=1e4)


def test_build_galaxy_clustering_measurement_set(tmp_path):
    view = synthetic_point_view(tmp_path, n=5000, seed=8)
    ms = build_galaxy_clustering(
        view, tracer="gal", z_range=(0.3, 0.7),
        weights=[_fkp(), ColumnWeight("weight_comp")],
        nside_window=64, nside_region=4,
        nz_edges=np.linspace(0.0, 1.2, 25),
        randoms="generate", n_randoms=20000, seed=1,
    )
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["gal"]
    assert ps.kind == "pointset"
    assert "weight" in ps.catalog.columns and (ps.catalog["weight"] > 0).all()
    assert ps.randoms is not None and len(ps.randoms) == 20000
    assert ps.nz.method == "spec_hist"
    assert ps.provenance.randoms_source == "generated"
    assert ms.spec.statistic == "pk_multipole"
    ms.check_invariants()
    assert len(ps.region_map) == len(ps.catalog)
    assert "region_id" in ps.catalog.columns and "region_id" in ps.randoms.columns


def test_build_galaxy_clustering_ingests_randoms_view(tmp_path):
    """The randoms=<DatasetView> branch ingests official randoms (provenance)."""
    data = synthetic_point_view(tmp_path, n=3000, seed=8, name="data")
    rand = synthetic_point_view(tmp_path, n=9000, seed=42, name="randoms")
    ms = build_galaxy_clustering(
        data, tracer="gal", z_range=(0.1, 1.0),
        weights=[ColumnWeight("weight_comp")],
        nz_edges=np.linspace(0.0, 1.2, 25),
        randoms=rand, nside_region=4)            # ingest, not generate
    ps = ms.products["gal"]
    assert ps.provenance.randoms_source == "ingested"
    # B4 fix: ingested randoms are z-filtered to the data z_range + weighted
    assert ps.randoms is not None and 0 < len(ps.randoms) <= 9000
    assert ps.randoms["z"].between(0.1, 1.0).all()
    assert (ps.randoms["weight"] == 1.0).all()
    assert "region_id" in ps.randoms.columns     # shared region applied
    ms.check_invariants()


def test_unknown_randoms_arg_raises(tmp_path):
    """A typo like randoms='generated' must error, not silently drop randoms."""
    view = synthetic_point_view(tmp_path, n=500, seed=8)
    with pytest.raises(ValueError, match="randoms must be"):
        build_galaxy_clustering(view, tracer="gal", z_range=(0.1, 1.0),
                                weights=[ColumnWeight("weight_comp")],
                                nz_edges=np.linspace(0, 1.2, 13),
                                randoms="generated")     # typo


def test_invariants_reject_cosmology(tmp_path):
    view = synthetic_point_view(tmp_path, n=1000, seed=8)
    ms = build_galaxy_clustering(view, tracer="gal", z_range=(0.1, 1.0),
                                 weights=[ColumnWeight("weight_comp")],
                                 nside_window=32, nside_region=2,
                                 nz_edges=np.linspace(0, 1.2, 13),
                                 randoms="generate", n_randoms=2000, seed=1)
    with pytest.raises(ValueError, match="cosmology"):
        ms.check_invariants(_inject_cosmology=True)
