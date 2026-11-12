import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.synthetic import write_synthetic_dataset, write_synthetic_sightlines
from scipy.stats import ks_2samp

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.combine.weights import InverseVarianceWeight
from oneuniverse.data import DatasetView
from oneuniverse.measure.atoms import Nz, Regions, Window
from oneuniverse.measure.pipeline import (build_pointset, build_sightlines, generate_randoms,
                                          randoms_from_view)
from oneuniverse.measure.product import CATALOG_CORE, Sightline


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "ds"
    write_synthetic_dataset(p, 50_000, 0, ra_range=(100, 160), dec_range=(-10, 30))
    return DatasetView.from_path(p)


def test_build_pointset_footprint_fallback_recorded(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=12, seed=1)
    assert prod.window.source == "footprint" and prod.provenance["window_source"] == "footprint" and prod.nz.method == "spec_hist"
    assert prod.catalog.num_rows > 0 and pc.min(prod.catalog["z"]).as_py() >= 0.3 and (prod.catalog["region_id"].to_numpy() >= 0).all()
    assert prod.randoms.num_rows == 10 * prod.n and prod.provenance["randoms_source"] == "generated"
    assert prod.catalog.column_names == list(CATALOG_CORE)
    pv = prod.provenance
    assert pv["dataset"] == "synthetic" and pv["version"] == list(view.version) and pv["z"] == [0.3, 1.0]
    assert pv["weight"] == "none" and pv["seed"] == 1 and pv["n"] == prod.n and pv["nz_method"] == "spec_hist"


def test_positions_round_trip(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), randoms="none")
    back = view.take(prod.catalog["partition"].to_numpy(), prod.catalog["offset"].to_numpy(), columns=["source_row", "z"])
    assert back["source_row"].to_pylist() == prod.catalog["source_row"].to_pylist()
    assert np.array_equal(back["z"].to_numpy(), prod.catalog["z"].to_numpy())


def test_build_pointset_uses_given_mask(view):
    m = np.zeros(12 * 64**2, np.float32); m[radec_to_pix(64, np.array([130.0]), np.array([10.0]))[0]] = 1.0
    prod = build_pointset(view, z=(0.1, 1.5), nz_edges=np.linspace(0.1, 1.5, 5), mask=Window.from_mask(m, nside=64), randoms="generate", regions=1)
    assert prod.window.source == "mask" and prod.catalog.num_rows < 500 and Window.from_mask(m, nside=64).contains(prod.randoms["ra"].to_numpy(), prod.randoms["dec"].to_numpy()).all()
    assert prod.provenance["window_source"] == "mask"


def test_randoms_follow_window_and_nz(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), random_factor=20, seed=2)
    assert prod.window.contains(prod.randoms["ra"].to_numpy(), prod.randoms["dec"].to_numpy()).all()
    h_d, _ = np.histogram(prod.catalog["z"].to_numpy(), prod.nz.edges, weights=prod.catalog["weight"].to_numpy()); h_r, _ = np.histogram(prod.randoms["z"].to_numpy(), prod.nz.edges)
    assert np.allclose(h_r / h_r.sum(), h_d / h_d.sum(), atol=0.02) and ks_2samp(prod.catalog["z"].to_numpy(), prod.randoms["z"].to_numpy()).pvalue > 1e-3


def test_weight_and_quality_and_keep_columns(view):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), weight=InverseVarianceWeight("z_err"), quality=("quality", ">=", 1), keep_columns=["mag"], randoms="none")
    assert prod.randoms is None and "mag" in prod.catalog.column_names and (prod.catalog["weight"].to_numpy() > 0).all() and prod.provenance["weight"].startswith("InverseVariance")
    assert prod.catalog.column_names == list(CATALOG_CORE) + ["mag"]
    assert prod.provenance["quality"] == ["quality", ">=", 1] and prod.provenance["keep_columns"] == ["mag"]
    assert prod.provenance["randoms_source"] == "none"
    full = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), randoms="none")
    assert prod.n < full.n
    q = view.take(prod.catalog["partition"].to_numpy(), prod.catalog["offset"].to_numpy(), columns=["quality"])["quality"].to_numpy()
    assert (q >= 1).all()


def test_empty_selection_raises(view):
    with pytest.raises(ValidationError, match="empty"): build_pointset(view, z=(5.0, 6.0), nz_edges=[5, 6])
    with pytest.raises(ValidationError, match="empty"):
        build_pointset(view, z=(0.3, 1.0), nz_edges=[0.3, 1.0], quality=("quality", ">", 99))


def test_nonfinite_rows_dropped(tmp_path):
    from fixtures.synthetic import synthetic_point_table
    from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
    from oneuniverse.data.schema import Geometry
    from oneuniverse.data.writer import write_dataset
    t = synthetic_point_table(2000, 3, ra_range=(100, 160), dec_range=(-10, 30))
    z = t["z"].to_numpy().copy(); z[:50] = np.nan
    t = t.set_column(t.column_names.index("z"), "z", pa.array(z, from_pandas=False))
    write_dataset(t, tmp_path / "d", geometry=Geometry.POINT, survey=SurveyMeta("s", "t"), loader=LoaderInfo("t", "1", {}))
    prod = build_pointset(DatasetView.from_path(tmp_path / "d"), z=(0.0, 3.0), nz_edges=[0.0, 3.0], randoms="none", regions=3)
    assert np.isfinite(prod.catalog["z"].to_numpy()).all() and prod.n <= 1950


def test_generate_randoms_exact_n_and_deterministic(view):
    w = Window.footprint(np.array([130.0, 131.0]), np.array([10.0, 11.0]), nside=64)
    r = Regions.from_window(w, 2, nside=32)
    nz = Nz(np.linspace(0.3, 1.0, 8), np.arange(1, 8), "spec_hist")
    a = generate_randoms(w, nz, r, n=1234, seed=5); b = generate_randoms(w, nz, r, n=1234, seed=5); c = generate_randoms(w, nz, r, n=1234, seed=6)
    assert a.num_rows == 1234 and a.equals(b) and not a.equals(c)
    assert a.column_names == ["ra", "dec", "z", "weight", "region_id"] and (a["weight"].to_numpy() == 1).all()
    assert w.contains(a["ra"].to_numpy(), a["dec"].to_numpy()).all() and (a["region_id"].to_numpy() >= 0).all()


def test_build_sightlines_from_sightline_view(tmp_path, view):
    write_synthetic_sightlines(tmp_path / "sl", 300, 0); v = DatasetView.from_path(tmp_path / "sl")
    sl = build_sightlines(v, z=(2.0, 2.5), regions=5); assert isinstance(sl, Sightline) and sl.n_pixels == sl.n * 50 and (sl.catalog["region_id"].to_numpy() >= 0).all()
    assert (sl.catalog["weight"].to_numpy() == 1.0).all() and "pixel_weight" in sl.catalog.column_names
    assert sl.catalog.column_names[:9] == ["sightline_id", "ra", "dec", "z", "weight", "region_id", "source_row", "partition", "offset"]
    with pytest.raises(FormatError): build_sightlines(view, z=(0.3, 1.0))


def test_randoms_from_view_ingest(view, tmp_path):
    prod = build_pointset(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), randoms=view)
    assert prod.provenance["randoms_source"] == "ingested" and prod.randoms.num_rows == prod.n and (prod.randoms["region_id"].to_numpy() >= 0).all()


def test_randoms_from_view_subsample(view):
    w = Window.footprint(np.array([130.0]), np.array([10.0]), nside=8)
    r = Regions.from_window(w, 1, nside=8)
    a = randoms_from_view(view, z=(0.3, 1.0), window=w, regions=r, n_max=100)
    b = randoms_from_view(view, z=(0.3, 1.0), window=w, regions=r, n_max=100)
    assert a.num_rows == 100 and a.equals(b) and (a["weight"].to_numpy() == 1).all()
