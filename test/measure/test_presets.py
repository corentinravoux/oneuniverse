import numpy as np
import pytest
from fixtures.synthetic import write_synthetic_dataset

from oneuniverse._core.errors import ValidationError
from oneuniverse.combine.weights import ShearWeight
from oneuniverse.data import DatasetView
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.presets import cosmic_shear, galaxy_clustering, three_by_two


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "ds"
    write_synthetic_dataset(p, 20_000, 0, ra_range=(100, 160), dec_range=(-10, 30))
    return DatasetView.from_path(p)


def _gc(view, **kw):
    return galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), **kw)


def test_galaxy_clustering_preset(view):
    ms = _gc(view)
    assert ms.spec.tracers == ("gal",) and ms.spec.statistics == ("pk_multipoles",)
    assert ms.products["gal"].randoms is not None


def test_cosmic_shear_preset(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(tmp_path / "s", 20_000, 0, with_pdf=spec, with_shear=True)
    v = DatasetView.from_path(tmp_path / "s")
    ms = cosmic_shear(v, z=(0.2, 2.0), shear_weight=ShearWeight("metacal"), tomo_edges=[0.2, 0.8, 2.0],
                      nz_edges=np.linspace(0, 3, 31), n_draws=5)
    src = ms.products["src"]
    assert {"e1", "e2", "tomo_bin", "z_draws"} <= set(src.extras)
    assert src.nz.counts.shape[0] == 2 and ms.spec.statistics == ("xi_plus", "xi_minus")
    assert src.extras["e1"].shape == (src.n,) and src.randoms is None


def test_three_by_two_requires_shared_regions(view):
    lens = _gc(view, regions=10)
    src = _gc(view, regions=10, name="src")
    with pytest.raises(ValidationError, match="regions"):
        three_by_two(lens, _gc(view, regions=12))
    src2 = MeasurementSet({"src": src.products["gal"]},
                          MeasurementSpec("src", ("src",), ("xi_plus",)), lens.regions)
    ms = three_by_two(lens, src2)
    assert ms.spec.pairs == (("gal", "gal"), ("gal", "src"), ("src", "src"))
    assert set(ms.products) == {"gal", "src"}
    assert ms.spec.statistics == ("pk_multipoles", "xi_plus")


def test_three_by_two_rejects_same_tracer_name(view):
    a = _gc(view, regions=10)
    with pytest.raises(ValidationError, match="differ"):
        three_by_two(a, a)
